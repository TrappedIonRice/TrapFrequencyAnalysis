"""
Export per-trap JSON bundles for the static inverse-control web app.

Usage (from repo root, with the repo's Python environment active):
    python export_bundles.py [--out static/bundles.js]

Writes: static/bundles.js  (window.TRAP_BUNDLES = {...})
        static/bundles.json (same data, raw JSON for debugging)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

# ── repo imports ─────────────────────────────────────────────────────────────
import constants
from control_constraints import build_L_b_for_point, build_target_hessian
from inverse_design import solve_u_for_targets
from linearsyssolve101 import build_voltage_to_c_matrix_cached
from trap_A_cache import DEFAULT_CACHE_DIR

# ── golden-test target definitions ───────────────────────────────────────────
# Three physically varied targets used for convention + solver validation.
# Frequencies in MHz; position in µm.
GOLDEN_TARGETS = [
    {
        "label": "origin_default",
        "r_um": [0.0, 0.0, 0.0],
        "principal_axis": [1.0, 0.0, 0.0],
        "ref_dir": [0.0, 1.0, 0.0],
        "alpha_deg": 0.0,
        "freqs_mhz": [0.2, 1.0, 1.0],
    },
    {
        "label": "shifted_x",
        "r_um": [10.0, 0.0, 0.0],
        "principal_axis": [1.0, 0.0, 0.0],
        "ref_dir": [0.0, 1.0, 0.0],
        "alpha_deg": 0.0,
        "freqs_mhz": [0.3, 0.8, 1.2],
    },
    {
        "label": "alpha_rotated",
        "r_um": [5.0, 0.0, 0.0],
        "principal_axis": [1.0, 0.0, 0.0],
        "ref_dir": [0.0, 1.0, 0.0],
        "alpha_deg": 45.0,
        "freqs_mhz": [0.2, 0.9, 1.1],
    },
]

# Weight on s in the app's W = diag(1,...,1, W_S).
# This differs from l2_sx100 (w_s=100). See plan notes.
W_S = 10.0

# Voltage bounds used both for A-matrix construction and QP solving.
DC_BOUNDS = (-500.0, 500.0)
RF_DC_BOUNDS = (-50.0, 50.0)
# Use 1500V max RF amplitude to accommodate traps with high s requirements.
# InnTrapFine needs s≈25.75 for (0.2,1.0,1.0) MHz; RF_S_MAX_DEFAULT (600V) gives only s≈4.9.
_RF_AMP_MAX_V = 1500.0
_OMEGA_MHZ = 2.0 * 3.141592653589793 * (constants.RF_FREQ_REF_HZ / 1e6)
S_BOUNDS = (0.0, (_RF_AMP_MAX_V ** 2) / (_OMEGA_MHZ ** 2))
POLYFIT_DEG = 4
NUM_SAMPLES = 200
SEED = 0


def build_golden_test(
    target: Dict,
    A: np.ndarray,
    powers: np.ndarray,
    dc_electrodes: List[str],
    rf_dc_electrodes: List[str],
    trap_name: str,
) -> Dict:
    """Compute L, b, M, and solved u for one golden target."""
    r_um = np.array(target["r_um"], dtype=float)
    r_m = r_um * 1e-6
    principal_axis = np.array(target["principal_axis"], dtype=float)
    ref_dir = np.array(target["ref_dir"], dtype=float)
    alpha_deg = float(target["alpha_deg"])
    freqs_hz = np.array(target["freqs_mhz"], dtype=float) * 1e6

    Kstar = build_target_hessian(
        freqs_hz,
        principal_axis,
        ref_dir,
        alpha_deg,
        mass=constants.ion_mass,
        charge=constants.ion_charge,
        poly_is_potential_energy=False,
        freqs_in_hz=True,
    )
    L, b = build_L_b_for_point(
        powers,
        r_m,
        Kstar,
        include_gradient=True,
        basis="nondim",
        nd_L0_m=constants.ND_L0_M,
    )
    M_mat = L @ A  # 9 × n_controls

    n_dc = len(dc_electrodes)
    n_rf = len(rf_dc_electrodes)
    u_bounds = (
        [DC_BOUNDS] * n_dc
        + [RF_DC_BOUNDS] * n_rf
        + [S_BOUNDS]
    )

    result = solve_u_for_targets(
        r0=r_m,
        freqs=freqs_hz,
        target_mode="exact",
        principal_axis=principal_axis,
        ref_dir=ref_dir,
        alpha_deg=alpha_deg,
        ion_mass_kg=constants.ion_mass,
        ion_charge_c=constants.ion_charge,
        poly_is_potential_energy=False,
        freqs_in_hz=True,
        trap_name=trap_name,
        dc_electrodes=dc_electrodes,
        rf_dc_electrodes=rf_dc_electrodes,
        num_samples=NUM_SAMPLES,
        dc_bounds=DC_BOUNDS,
        rf_dc_bounds=RF_DC_BOUNDS,
        s_bounds=S_BOUNDS,
        polyfit_deg=POLYFIT_DEG,
        seed=SEED,
        use_cache=True,
        objective="weighted_l2",
        s_penalty_scale=W_S,
        enforce_bounds=True,
        u_bounds=u_bounds,
    )

    status = result["status"]
    u_solved = result["u"]
    if u_solved is None:
        print(
            f"  WARNING: solve failed for target '{target['label']}' "
            f"(status={status}). Storing None."
        )
        u_solved_list = None
    else:
        u_solved_list = u_solved.tolist()

    return {
        "label": target["label"],
        "r_um": r_um.tolist(),
        "principal_axis": principal_axis.tolist(),
        "ref_dir": ref_dir.tolist(),
        "alpha_deg": alpha_deg,
        "freqs_hz": freqs_hz.tolist(),
        "L": L.tolist(),
        "b": b.tolist(),
        "M": M_mat.tolist(),
        "u_solved": u_solved_list,
        "solve_status": status,
        "w_s": W_S,
    }


def export_trap(registry_key: str, cfg: Dict) -> Dict:
    """Build the full JSON bundle for one trap."""
    trap_name = cfg["trap_name"]
    dc_electrodes = list(cfg["dc_electrodes"])
    rf_dc_electrodes = list(cfg["rf_dc_electrodes"])
    n_dc = len(dc_electrodes)
    n_rf = len(rf_dc_electrodes)
    n_controls = n_dc + n_rf + 1  # +1 for s

    print(f"\n=== Exporting {registry_key} (trap_name={trap_name}, "
          f"n_controls={n_controls}) ===")

    # Build / load A matrix from cache
    cache_out = build_voltage_to_c_matrix_cached(
        trap_name=trap_name,
        dc_electrodes=dc_electrodes,
        rf_dc_electrodes=rf_dc_electrodes,
        num_samples=NUM_SAMPLES,
        dc_bounds=[DC_BOUNDS] * n_dc,
        rf_dc_bounds=[RF_DC_BOUNDS] * n_rf,
        s_bounds=S_BOUNDS,
        polyfit_deg=POLYFIT_DEG,
        seed=SEED,
        cache_dir=DEFAULT_CACHE_DIR,
        ion_mass_kg=constants.ion_mass,
    )
    A = np.asarray(cache_out["A"], dtype=float)
    powers = np.asarray(cache_out["powers"], dtype=int)
    cache_hit = cache_out.get("cache_hit", False)
    print(f"  A shape: {A.shape}  cache_hit={cache_hit}")

    # Build golden tests
    golden_tests = []
    for tgt in GOLDEN_TARGETS:
        print(f"  Golden target: {tgt['label']} ...", end="", flush=True)
        gt = build_golden_test(
            tgt, A, powers, dc_electrodes, rf_dc_electrodes, trap_name
        )
        golden_tests.append(gt)
        print(f" status={gt['solve_status']}")

    bundle = {
        "registry_key": registry_key,
        "display_label": cfg["display_label"],
        "trap_name": trap_name,
        "dc_electrodes": dc_electrodes,
        "rf_dc_electrodes": rf_dc_electrodes,
        "n_controls": n_controls,
        "control_names": dc_electrodes + rf_dc_electrodes + ["s"],
        "A": A.tolist(),
        "powers": powers.tolist(),
        "ion_mass": float(constants.ion_mass),
        "ion_charge": float(constants.ion_charge),
        "ND_L0_M": float(constants.ND_L0_M),
        "RF_FREQ_REF_HZ": float(constants.RF_FREQ_REF_HZ),
        "RF_OMEGA_REF_MHZ": float(constants.RF_OMEGA_REF_MHZ),
        "RF_S_MAX_DEFAULT": float(constants.RF_S_MAX_DEFAULT),
        "polyfit_deg": POLYFIT_DEG,
        "default_dc_bounds": list(DC_BOUNDS),
        "default_rf_dc_bounds": list(RF_DC_BOUNDS),
        "default_s_bounds": list(S_BOUNDS),
        "rf_amp_max_v": _RF_AMP_MAX_V,
        "fit_box_um": {
            "x": float(constants.center_region_x_um),
            "y": float(constants.center_region_y_um),
            "z": float(constants.center_region_z_um),
        },
        "app_w_s": W_S,
        "golden_tests": golden_tests,
    }
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trap bundles for the web app.")
    parser.add_argument(
        "--out", default="static/bundles.js", help="Output .js path"
    )
    args = parser.parse_args()

    out_js = args.out
    out_json = out_js.replace(".js", ".json")
    os.makedirs(os.path.dirname(out_js) or ".", exist_ok=True)

    all_bundles: Dict[str, Dict] = {}
    for registry_key, cfg in constants.INVERSE_APP_TRAP_REGISTRY.items():
        try:
            bundle = export_trap(registry_key, cfg)
            all_bundles[registry_key] = bundle
        except Exception as exc:
            print(f"  ERROR exporting {registry_key}: {exc}", file=sys.stderr)
            raise

    # Write raw JSON (debug)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_bundles, f, indent=2)
    print(f"\nWrote {out_json}")

    # Write JS with global assignment
    js_payload = json.dumps(all_bundles, separators=(",", ":"))
    with open(out_js, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by export_bundles.py — do not edit by hand.\n")
        f.write("window.TRAP_BUNDLES = ")
        f.write(js_payload)
        f.write(";\n")
    print(f"Wrote {out_js}")

    # Summary
    print("\nSummary:")
    for key, b in all_bundles.items():
        n_ok = sum(1 for gt in b["golden_tests"] if gt["solve_status"] == "ok")
        print(f"  {key}: A={np.array(b['A']).shape}, "
              f"golden={n_ok}/{len(b['golden_tests'])} ok")


if __name__ == "__main__":
    main()
