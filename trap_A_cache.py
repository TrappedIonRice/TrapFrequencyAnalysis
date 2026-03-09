"""
Persistent cache for A matrix (and powers/metadata) keyed by trap config.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Tuple
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

DEFAULT_CACHE_DIR = os.path.join(".cache", "trap_A")
_CACHE_VERSION = "A_cache_v5"


def normalize_bounds(bounds: Iterable[Tuple[float, float]], n: int) -> List[List[float]]:
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2 and not isinstance(bounds[0], (list, tuple)):
        lo, hi = bounds
        return [[float(lo), float(hi)] for _ in range(n)]
    if isinstance(bounds, (list, tuple)) and len(bounds) == n:
        return [[float(b[0]), float(b[1])] for b in bounds]
    raise ValueError("bounds must be (low, high) or a list of (low, high) per electrode")


def make_cache_key(cfg_dict: Dict[str, Any]) -> str:
    cfg_json = json.dumps(cfg_dict, sort_keys=True)
    h = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()
    return h


def cache_file_path(cache_dir: str, trap_name: str, polyfit_deg: int, key: str) -> str:
    subdir = os.path.join(cache_dir, trap_name, f"deg{polyfit_deg}")
    os.makedirs(subdir, exist_ok=True)
    return os.path.join(subdir, f"{key[:12]}.npz")


def load_A_from_cache(path: str) -> Dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as data:
        A = data["A"]
        powers = data["powers"]
        cfg_json = str(data["cfg_json"].item())
        version = str(data["version"].item())
        created_utc = str(data["created_utc"].item())

        out: Dict[str, Any] = {
            "A": A,
            "powers": powers,
            "cfg": json.loads(cfg_json),
            "version": version,
            "created_utc": created_utc,
            "path": path,
        }
        if "cfg_full_json" in data.files:
            out["cfg_full"] = json.loads(str(data["cfg_full_json"].item()))
        # Optional extras
        for k in [
            "u_layout",
            "rf_dc_electrodes",
            "fit_rel_rmse",
            "fit_r2_per_c",
            "nonlinearity_flags",
        ]:
            if k in data.files:
                out[k] = data[k]
        return out


def save_A_to_cache(
    path: str,
    A: np.ndarray,
    powers: np.ndarray,
    cfg_key: Dict[str, Any],
    cfg_full: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> str:
    cfg_json = json.dumps(cfg_key, sort_keys=True)
    created_utc = datetime.now(timezone.utc).isoformat()
    payload = {
        "A": np.asarray(A, dtype=float),
        "powers": np.asarray(powers, dtype=int),
        "cfg_json": np.array(cfg_json),
        "version": np.array(_CACHE_VERSION),
        "created_utc": np.array(created_utc),
    }
    if cfg_full is not None:
        payload["cfg_full_json"] = np.array(json.dumps(cfg_full, sort_keys=True))
    if extra:
        for k, v in extra.items():
            payload[k] = v
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp.npz"
    np.savez(tmp_path, **payload)
    os.replace(tmp_path, path)
    return created_utc


def _index_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "index.json")


def load_index(cache_dir: str) -> Dict[str, Any]:
    path = _index_path(cache_dir)
    if not os.path.exists(path):
        return {"schema_version": "1", "entries": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_index_atomic(cache_dir: str, index_obj: Dict[str, Any]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = _index_path(cache_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index_obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def upsert_index_entry(cache_dir: str, entry: Dict[str, Any]) -> None:
    idx = load_index(cache_dir)
    entries = idx.get("entries", [])
    replaced = False
    for i, existing in enumerate(entries):
        if existing.get("key") == entry.get("key"):
            entries[i] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)
    idx["entries"] = entries
    write_index_atomic(cache_dir, idx)


def get_or_build_A(
    *,
    cache_dir: str,
    trap_name: str,
    dc_electrodes: List[str],
    rf_dc_electrodes: List[str],
    polyfit_deg: int,
    dc_bounds: Iterable[Tuple[float, float]],
    rf_dc_bounds: Iterable[Tuple[float, float]],
    s_bounds: Tuple[float, float],
    num_samples: int,
    seed: int,
    ion_mass_kg: float | None = None,
    builder_fn: Callable[..., Dict[str, Any]],
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    import constants

    cfg_full = {
        "trap_name": trap_name,
        "dc_electrodes": list(dc_electrodes),
        "rf_dc_electrodes": list(rf_dc_electrodes),
        "polyfit_deg": int(polyfit_deg),
        "dc_bounds": normalize_bounds(dc_bounds, len(dc_electrodes)),
        "rf_dc_bounds": normalize_bounds(rf_dc_bounds, len(rf_dc_electrodes)),
        "s_bounds": [float(s_bounds[0]), float(s_bounds[1])],
        "num_samples": int(num_samples),
        "seed": int(seed),
        "basis": "nondim",
        "nd_L0_m": float(constants.ND_L0_M),
        "center_region_x_um": float(constants.center_region_x_um),
        "center_region_y_um": float(constants.center_region_y_um),
        "center_region_z_um": float(constants.center_region_z_um),
        "version": _CACHE_VERSION,
    }
    if ion_mass_kg is not None:
        cfg_full["ion_mass_kg"] = float(ion_mass_kg)

    # Key should ignore sampling/bounds so changes there do not invalidate A.
    cfg_key = {
        "trap_name": trap_name,
        "dc_electrodes": list(dc_electrodes),
        "rf_dc_electrodes": list(rf_dc_electrodes),
        "polyfit_deg": int(polyfit_deg),
        "basis": "nondim",
        "nd_L0_m": float(constants.ND_L0_M),
        "center_region_x_um": float(constants.center_region_x_um),
        "center_region_y_um": float(constants.center_region_y_um),
        "center_region_z_um": float(constants.center_region_z_um),
        "version": _CACHE_VERSION,
    }
    if ion_mass_kg is not None:
        cfg_key["ion_mass_kg"] = float(ion_mass_kg)

    key = make_cache_key(cfg_key)
    path = cache_file_path(cache_dir, trap_name, polyfit_deg, key)

    if not force_rebuild:
        cached = load_A_from_cache(path)
        if cached is not None and cached.get("cfg") == cfg_key:
            cached["cache_hit"] = True
            cached["cache_path"] = path
            cached["cfg"] = cfg_full
            return cached

    out = builder_fn(
        trap_name=trap_name,
        dc_electrodes=dc_electrodes,
        rf_dc_electrodes=rf_dc_electrodes,
        num_samples=num_samples,
        dc_bounds=dc_bounds,
        rf_dc_bounds=rf_dc_bounds,
        s_bounds=s_bounds,
        polyfit_deg=polyfit_deg,
        seed=seed,
    )
    extra = {
        "u_layout": np.array(json.dumps(out.get("u_layout", {}))),
        "rf_dc_electrodes": np.array(json.dumps(out.get("rf_dc_electrodes", []))),
        "fit_rel_rmse": np.array(out.get("fit_rel_rmse", np.nan)),
        "fit_r2_per_c": np.array(out.get("fit_r2_per_c", np.array([]))),
        "nonlinearity_flags": np.array(json.dumps(out.get("nonlinearity_flags", []))),
    }
    created_utc = save_A_to_cache(path, out["A"], out["powers"], cfg_key, cfg_full=cfg_full, extra=extra)
    rel_path = os.path.relpath(path, cache_dir)
    upsert_index_entry(
        cache_dir,
        {
            "key": key,
            "key12": key[:12],
            "trap_name": trap_name,
            "polyfit_deg": int(polyfit_deg),
            "path": rel_path,
            "created_utc": created_utc,
            "cfg": cfg_full,
            "cfg_key": cfg_key,
            "fit_rel_rmse": out.get("fit_rel_rmse"),
        },
    )
    out["cache_hit"] = False
    out["cache_path"] = path
    out["cfg"] = cfg_full
    out["cfg_key"] = cfg_key
    return out
