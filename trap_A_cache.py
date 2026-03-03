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
_CACHE_VERSION = "A_cache_v1"


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
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{trap_name}_deg{polyfit_deg}_{key[:12]}.npz")


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
    cfg_dict: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> None:
    cfg_json = json.dumps(cfg_dict, sort_keys=True)
    payload = {
        "A": np.asarray(A, dtype=float),
        "powers": np.asarray(powers, dtype=int),
        "cfg_json": np.array(cfg_json),
        "version": np.array(_CACHE_VERSION),
        "created_utc": np.array(datetime.now(timezone.utc).isoformat()),
    }
    if extra:
        for k, v in extra.items():
            payload[k] = v
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **payload)


def get_or_build_A(
    *,
    cache_dir: str,
    trap_name: str,
    dc_electrodes: List[str],
    rf_dc_electrodes: List[str],
    rf_freq_hz: float,
    polyfit_deg: int,
    dc_bounds: Iterable[Tuple[float, float]],
    rf_dc_bounds: Iterable[Tuple[float, float]],
    rf2_bounds: Tuple[float, float],
    num_samples: int,
    seed: int,
    builder_fn: Callable[..., Dict[str, Any]],
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    cfg = {
        "trap_name": trap_name,
        "dc_electrodes": list(dc_electrodes),
        "rf_dc_electrodes": list(rf_dc_electrodes),
        "rf_freq_hz": float(rf_freq_hz),
        "polyfit_deg": int(polyfit_deg),
        "dc_bounds": normalize_bounds(dc_bounds, len(dc_electrodes)),
        "rf_dc_bounds": normalize_bounds(rf_dc_bounds, len(rf_dc_electrodes)),
        "rf2_bounds": [float(rf2_bounds[0]), float(rf2_bounds[1])],
        "num_samples": int(num_samples),
        "seed": int(seed),
        "version": _CACHE_VERSION,
    }
    key = make_cache_key(cfg)
    path = cache_file_path(cache_dir, trap_name, polyfit_deg, key)

    if not force_rebuild:
        cached = load_A_from_cache(path)
        if cached is not None and cached.get("cfg") == cfg:
            cached["cache_hit"] = True
            cached["cache_path"] = path
            return cached

    out = builder_fn(
        trap_name=trap_name,
        dc_electrodes=dc_electrodes,
        rf_dc_electrodes=rf_dc_electrodes,
        rf_freq_hz=rf_freq_hz,
        num_samples=num_samples,
        dc_bounds=dc_bounds,
        rf_dc_bounds=rf_dc_bounds,
        rf2_bounds=rf2_bounds,
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
    save_A_to_cache(path, out["A"], out["powers"], cfg, extra=extra)
    out["cache_hit"] = False
    out["cache_path"] = path
    out["cfg"] = cfg
    return out

