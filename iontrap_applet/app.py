import math
import os
import sys
import json
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from io import BytesIO


import numpy as np
import pandas as pd

from BasicFinderLinearDrivenG02F110 import (
    calculate_jacobian_F110,
    solve_F110_for_targets,
)

from BasicFinderLinearDrivenG03F310 import (
    find_max_coupling_tensor_F310,
    solve_F310_for_targets as solve_F310_for_targets_tensor,
)

from BasicFinderLinearDrivenG02F120 import (
    find_max_coupling_matrix_F120,
    solve_F120_for_targets,
)

from BasicFinderLinearDrivenG03F320 import (
    find_max_coupling_matrix_F320,
    solve_F320_for_targets,
)


try:
    from matplotlib.colors import LinearSegmentedColormap

    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    from sim.equilibrium_viewer import (
        build_equilibrium_figure,
        build_equilibrium_summary_lines,
    )
except Exception:
    build_equilibrium_figure = None
    build_equilibrium_summary_lines = None


# Persist last results and config across reruns
if "res" not in st.session_state:
    st.session_state["res"] = None
if "cfg" not in st.session_state:
    st.session_state["cfg"] = None


import itertools


def fig_to_png_bytes(fig) -> bytes:
    """Convert a matplotlib Figure to PNG bytes (for storing in cached results)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _symmetrize_rank3_abs(G: np.ndarray) -> np.ndarray:
    """
    Enforce full permutation symmetry on a rank-3 tensor by setting every
    permutation of (a,b,c) to the SAME nonnegative magnitude (max |val| across the group).
    This fills diagonals like (a,a,c) from entries such as (a,c,a)/(c,a,a), etc.
    """
    K = G.shape[0]
    out = G.copy()
    seen = set()
    for a in range(K):
        for b in range(K):
            for c in range(K):
                key = tuple(sorted((a, b, c)))
                if key in seen:
                    continue
                seen.add(key)
                perms = set(itertools.permutations((a, b, c), 3))
                vals = [abs(out[i, j, k]) for (i, j, k) in perms]
                m = max(vals)
                for i, j, k in perms:
                    out[i, j, k] = m
    return out


def _symmetrize_rank3_equal(G: np.ndarray) -> np.ndarray:
    """
    Enforce full permutation symmetry by assigning every permutation of (a,b,c)
    to the same value. We choose the representative as the entry with the largest |val|
    among the permutations (sign preserved). Suitable for predicted tensors.
    """
    K = G.shape[0]
    out = G.copy()
    seen = set()
    for a in range(K):
        for b in range(K):
            for c in range(K):
                key = tuple(sorted((a, b, c)))
                if key in seen:
                    continue
                seen.add(key)
                perms = set(itertools.permutations((a, b, c), 3))
                vals = [out[i, j, k] for (i, j, k) in perms]
                # pick representative with max absolute magnitude (keep sign)
                idx = int(np.argmax(np.abs(vals)))
                rep = float(vals[idx])
                for i, j, k in perms:
                    out[i, j, k] = rep
    return out


# --- Principal-direction color palette (consistent across the app) ---
DIR_COLORS = {
    0: "#1f77b4",  # dir_0  (lowest 1-ion freq)
    1: "#ff7f0e",  # dir_1
    2: "#2ca02c",  # dir_2
}


def dir_color(idx: int) -> str:
    return DIR_COLORS.get(int(idx), "#888888")


def _independent_channel_order(preset: str) -> List[str]:
    """
    Channel order used by F120/F320 solvers.

    Legacy: 20 channels = DC1..DC10 + RF11..RF20
    twodTrap_1: 12 channels = DC1..DC12
    """
    p = str(preset).strip()
    if p == "twodTrap_1":
        return [f"DC{i}" for i in range(1, 13)]
    return [f"DC{i}" for i in range(1, 11)] + [f"RF{i}" for i in range(11, 21)]


def _to_jsonable(obj):
    import numpy as np, json

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _eq_from_ion_equilibrium_positions(sim, num_ions):
    """
    Return sim.ion_equilibrium_positions[num_ions] if it exists and is (N,3).
    Does not call any compute methods.
    """
    import numpy as np

    d = getattr(sim, "ion_equilibrium_positions", None)
    if isinstance(d, dict) and (num_ions in d):
        arr = np.asarray(d[num_ions], dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
    return None


def _eq_metadata_from_sim(sim, num_ions, minimizer_type):
    if hasattr(sim, "get_equilibrium_metadata"):
        try:
            metadata = sim.get_equilibrium_metadata(num_ions, minimizer_type)
        except TypeError:
            metadata = sim.get_equilibrium_metadata(
                num_ions,
                minimizertype=minimizer_type,
            )
        if isinstance(metadata, dict):
            return metadata

    metadata_store = getattr(sim, "equilibrium_metadata", None)
    if isinstance(metadata_store, dict):
        bucket = metadata_store.get(num_ions)
        if isinstance(bucket, dict):
            metadata = bucket.get(minimizer_type)
            if isinstance(metadata, dict):
                return metadata
    return None


def compute_f110_max_matrix(tv, num_ions, preset, point=None, bounds=None):
    """
    Compute per-pair max g0 (Hz) within box bounds for the 5 symmetric inputs,
    and return a 3N×3N matrix with only the upper triangle filled.
    """
    if point is None:
        point = [0.0, 0.0, 0.0, 0.0, 0.0]
    if bounds is None:
        bounds = [(-1.0, 1.0)] * 5

    out = calculate_jacobian_F110(
        num_ions=int(num_ions),
        constant_trappingvars=tv,  # must already include twist/endcaps, RF, DC, etc.
        point=point,
        simulation_preset=preset,
    )
    J = out["J_Hz_per_V"]  # (M×5) Hz/V
    g_center = out["g_center_Hz"]  # (M,) Hz
    pairs = out["pairs"]  # list[(i,j)]
    K = out["K"]

    # c = g(point) - J @ point
    pvec = np.asarray(point, float)
    c_vec = g_center - J @ pvec

    # row-wise box maximization of J u + c over bounds
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    choose = (J >= 0.0).astype(float)  # (M×5)
    u_star = choose * hi + (1.0 - choose) * lo
    g_max_vec = (J * u_star).sum(axis=1) + c_vec

    # fill upper triangle
    Mmat = np.zeros((K, K), float)
    for idx, (i, j) in enumerate(pairs):
        Mmat[i, j] = g_max_vec[idx]
    return Mmat


# ------------------------------------------------------------
# App configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Ion Trap Resonance Explorer", layout="wide")

st.title("🔬 Ion Trap Resonance Explorer")
st.caption(
    "Quick UI to tweak trapping variables, choose ion count, and inspect equilibria, modes, frequencies, and resonant couplings."
)

# ------------------------------------------------------------
# Sidebar inputs (as a FORM so edits don't apply until 'Compute')
# ------------------------------------------------------------
with st.sidebar:
    with st.form("params"):
        st.header("Trap / Simulation Setup")
        data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
        data_presets = []
        if os.path.isdir(data_root):
            data_presets = sorted(
                [
                    d
                    for d in os.listdir(data_root)
                    if os.path.isdir(os.path.join(data_root, d))
                ]
            )
        preset_options = data_presets + ["Custom"]

        preset = st.selectbox(
            "Simulation preset (name passed to Simulation)",
            options=preset_options,
            index=0 if preset_options else 0,
        )

        if preset == "Custom":
            preset = st.text_input("Custom preset string", value="Simp58_101")

        num_ions = st.number_input(
            "Number of ions", min_value=1, max_value=120, value=3, step=1
        )
        minimizer_type = st.selectbox(
            "Equilibrium minimizer",
            options=[
                "Normal",
                "InitGuess",
                "Dummy1",
                "Dummy2",
                "Dummy3",
                "Quartic2D_101",
                "cryo2dcopy",
                "Cryo2d_closecopy",
                "Cryo2d_closecopy_2",
            ],
            index=0,
        )
        poly_deg = st.selectbox(
            "Polynomial degree for fits", options=[2, 3, 4, 5, 6], index=2
        )
        generate_voltage_plots = st.checkbox(
            "Generate_Voltage_Visualization_Plots", value=False
        )
        compute_resonant_couplings = st.checkbox(
            "Compute resonant coupolings", value=False
        )
        generate_mm_plot = st.checkbox(
            "Generate_Micromotion_Plot", value=False
        )

        st.subheader("RF Drive")
        rf_freq = st.number_input(
            "RF frequency (Hz)",
            min_value=1.0,
            value=25_500_000.0,
            step=1_000.0,
            format="%.3f",
        )
        rf_amp1 = st.number_input("RF1 amplitude (V)", value=377.0, step=0.5)
        rf_amp2 = st.number_input("RF2 amplitude (V)", value=377.0, step=0.5)

        st.subheader("Extra Drive (optional)")
        use_extra = st.checkbox("Enable Extra Drive", value=True)
        extra_freq = st.number_input(
            "Extra Drive frequency (Hz)",
            min_value=0.0,
            value=250_000.0,
            step=100.0,
            format="%.3f",
        )
        default_extra_map = {
            "DC1": 0.175,
            "DC2": 0.060,
            "DC3": 0.0,
            "DC4": -0.60,
            "DC5": -0.175,
            "DC10": 0.175,
            "DC9": 0.060,
            "DC8": 0.0,
            "DC7": -0.60,
            "DC6": -0.175,
            "DC11": 0.0,
            "DC12": 0.0,
        }
        extra_map_json = st.text_area(
            "Extra Drive DC map (JSON dict)",
            value=json.dumps(default_extra_map, indent=2),
            height=160,
            help="Map electrode names to amplitudes (in volts). Example keys: DC1..DC12.",
        )

        with st.expander("DC Geometry (twist + endcaps)", expanded=False):
            apply_twist_endcaps = st.checkbox(
                "Enable twist/endcaps", value=True, key="apply_twist_endcaps"
            )
            twist = st.number_input(
                "apply_dc_twist_endcaps: twist", value=0.0, step=0.01
            )
            endcaps = st.number_input(
                "apply_dc_twist_endcaps: endcaps",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=0.1,
                format="%.3f",
            )

        if False:
            with st.expander(
                "Manual DC offsets (added on DC drive before twist/endcaps)",
                expanded=False,
            ):
                manual_dc_enabled = st.checkbox(
                    "Enable manual DC offsets", value=True, key="manual_dc_enabled"
                )

        # # keep a stable dict in session state
        # if "manual_dc" not in st.session_state:
        #     st.session_state["manual_dc"] = {
        #         k: 0.0
        #         for k in [
        #             "DC1",
        #             "DC2",
        #             "DC3",
        #             "DC4",
        #             "DC5",
        #             "DC6",
        #             "DC7",
        #             "DC8",
        #             "DC9",
        #             "DC10",
        #             "RF1",
        #             "RF2",
        #         ]
        #     }

        # keep a stable dict in session state
        if "manual_dc" not in st.session_state:
            st.session_state["manual_dc"] = {
                k: 0.0
                for k in [
                    # DC blades
                    "DC1",
                    "DC2",
                    "DC3",
                    "DC4",
                    "DC5",
                    "DC6",
                    "DC7",
                    "DC8",
                    "DC9",
                    "DC10",
                    "DC11",
                    "DC12",
                    # legacy per-blade RF (keep available)
                    "RF1",
                    "RF2",
                    # NEW: RF1 segments (left→right: RF11..RF15)
                    "RF11",
                    "RF12",
                    "RF13",
                    "RF14",
                    "RF15",
                    # NEW: RF2 segments (left→right by your diagram: RF20..RF16)
                    "RF20",
                    "RF19",
                    "RF18",
                    "RF17",
                    "RF16",
                ]
            }
        # """
        #     if manual_dc_enabled:
        #         # first row: DC5  DC4  DC3  DC2  DC1  RF1
        #         # Row 1: DC6 DC5 DC4 DC3 DC2 DC1 RF1
        #         cols_top = st.columns(7)
        #         order_top = ["DC6", "DC5", "DC4", "DC3", "DC2", "DC1", "RF1"]
        #         for col, key in zip(cols_top, order_top):
        #             st.session_state["manual_dc"][key] = col.number_input(
        #                 f"{key} (V)",
        #                 value=float(st.session_state["manual_dc"][key]),
        #                 step=0.01,
        #                 format="%.3f",
        #                 key=f"manual_dc_{key}",
        #             )

        #         # Row 2: DC7 DC8 DC9 DC10 DC11 DC12 RF2
        #         cols_bot = st.columns(7)
        #         order_bot = ["DC7", "DC8", "DC9", "DC10", "DC11", "DC12", "RF2"]
        #         for col, key in zip(cols_bot, order_bot):
        #             st.session_state["manual_dc"][key] = col.number_input(
        #                 f"{key} (V)",
        #                 value=float(st.session_state["manual_dc"][key]),
        #                 step=0.01,
        #                 format="%.3f",
        #                 key=f"manual_dc_{key}",
        #             )
        #         # third row: RF1 segments (left→right: RF11 RF12 RF13 RF14 RF15)
        #         cols_rf1 = st.columns(5)
        #         order_rf1 = ["RF11", "RF12", "RF13", "RF14", "RF15"]
        #         for col, key in zip(cols_rf1, order_rf1):
        #             st.session_state["manual_dc"][key] = col.number_input(
        #                 f"{key} (V)",
        #                 value=float(st.session_state["manual_dc"][key]),
        #                 step=0.01,
        #                 format="%.3f",
        #                 key=f"manual_dc_{key}",
        #             )

        #         # fourth row: RF2 segments (left→right by your diagram: RF20 RF19 RF18 RF17 RF16)
        #         cols_rf2 = st.columns(5)
        #         order_rf2 = ["RF20", "RF19", "RF18", "RF17", "RF16"]
        #         for col, key in zip(cols_rf2, order_rf2):
        #             st.session_state["manual_dc"][key] = col.number_input(
        #                 f"{key} (V)",
        #                 value=float(st.session_state["manual_dc"][key]),
        #                 step=0.01,
        #                 format="%.3f",
        #                 key=f"manual_dc_{key}",
        #             )

        # """
        with st.expander(
            "Manual DC offsets (added on DC drive before twist/endcaps)",
            expanded=False,
        ):
            manual_dc_enabled = st.checkbox(
                "Enable manual DC offsets", value=True, key="manual_dc_enabled"
            )

            if "manual_dc" not in st.session_state:
                st.session_state["manual_dc"] = {
                    k: 0.0
                    for k in [
                        "DC1",
                        "DC2",
                        "DC3",
                        "DC4",
                        "DC5",
                        "DC6",
                        "DC7",
                        "DC8",
                        "DC9",
                        "DC10",
                        "DC11",
                        "DC12",
                        "RF1",
                        "RF2",
                        "RF11",
                        "RF12",
                        "RF13",
                        "RF14",
                        "RF15",
                        "RF20",
                        "RF19",
                        "RF18",
                        "RF17",
                        "RF16",
                    ]
                }

            if manual_dc_enabled:
                cols_top = st.columns(7)
                order_top = ["DC6", "DC5", "DC4", "DC3", "DC2", "DC1", "RF1"]
                for col, key in zip(cols_top, order_top):
                    st.session_state["manual_dc"][key] = col.number_input(
                        f"{key} (V)",
                        value=float(st.session_state["manual_dc"][key]),
                        step=0.01,
                        format="%.3f",
                    )

                cols_bot = st.columns(7)
                order_bot = ["DC7", "DC8", "DC9", "DC10", "DC11", "DC12", "RF2"]
                for col, key in zip(cols_bot, order_bot):
                    st.session_state["manual_dc"][key] = col.number_input(
                        f"{key} (V)",
                        value=float(st.session_state["manual_dc"][key]),
                        step=0.01,
                        format="%.3f",
                    )

                cols_rf1 = st.columns(5)
                order_rf1 = ["RF11", "RF12", "RF13", "RF14", "RF15"]
                for col, key in zip(cols_rf1, order_rf1):
                    st.session_state["manual_dc"][key] = col.number_input(
                        f"{key} (V)",
                        value=float(st.session_state["manual_dc"][key]),
                        step=0.01,
                        format="%.3f",
                    )

                cols_rf2 = st.columns(5)
                order_rf2 = ["RF20", "RF19", "RF18", "RF17", "RF16"]
                for col, key in zip(cols_rf2, order_rf2):
                    st.session_state["manual_dc"][key] = col.number_input(
                        f"{key} (V)",
                        value=float(st.session_state["manual_dc"][key]),
                        step=0.01,
                        format="%.3f",
                    )

        with st.expander("2D DC layout manual offsets", expanded=False):
            manual_dc_2d_enabled = st.checkbox(
                "Enable 2D DC layout offsets", value=True, key="manual_dc_2d_enabled"
            )

            st.subheader("2D DC layout presets")
            dc_3e_endcaps = st.number_input(
                "3E endcaps (V)", value=0.0, step=0.01, format="%.3f"
            )
            dc_3e_center = st.number_input(
                "3E center (V)", value=0.0, step=0.01, format="%.3f"
            )
            dc_5e_endcaps = st.number_input(
                "5E endcaps (V)", value=0.0, step=0.01, format="%.3f"
            )
            dc_5e_mid = st.number_input(
                "5E mid (V)", value=0.0, step=0.01, format="%.3f"
            )
            dc_5e_center = st.number_input(
                "5E center (V)", value=0.0, step=0.01, format="%.3f"
            )

            if "manual_dc_2d" not in st.session_state:
                st.session_state["manual_dc_2d"] = {
                    k: 0.0 for k in [f"DC{i}" for i in range(1, 21)] + ["RF1", "RF2"]
                }

            if manual_dc_2d_enabled:
                rows = [
                    [f"DC{i}" for i in range(1, 6)],
                    [f"DC{i}" for i in range(6, 11)],
                    [f"DC{i}" for i in range(11, 16)],
                    [f"DC{i}" for i in range(16, 21)],
                    ["RF1", "RF2"],
                ]
                for row in rows:
                    cols = st.columns(len(row))
                    for col, key in zip(cols, row):
                        st.session_state["manual_dc_2d"][key] = col.number_input(
                            f"{key} (V)",
                            value=float(st.session_state["manual_dc_2d"][key]),
                            step=0.01,
                            format="%.3f",
                            key=f"manual_dc_2d_{key}",
                        )

        st.divider()
        st.header("Resonance Scan")
        tol_Hz = st.number_input(
            "Resonance tolerance ± (Hz)",
            min_value=0.0,
            value=1_000.0,
            step=10.0,
            format="%.3f",
        )
        orders_pick = st.multiselect(
            "Coupling orders", options=[2, 3, 4], default=[2, 3]
        )

        run_btn = st.form_submit_button("Compute", type="primary")

with st.sidebar:
    show_equilibrium_viewer = st.checkbox("Show equilibrium viewer", value=False)


# ------------------------------------------------------------
# Utilities: imports, hashing, and compute wrapper
# ------------------------------------------------------------
def _ensure_imports():
    """Ensure we can import Simulation and Trapping_Vars from the repo.
    Returns (Simulation, Trapping_Vars) types if successful; otherwise raises.
    """
    # Try a few plausible import paths, fall back gracefully.
    last_err = None
    for sim_mod in ("sim.simulation", "simulation", "Simulation", "simulation_fitting"):
        try:
            sim = __import__(sim_mod, fromlist=["Simulation"])
            Simulation = getattr(sim, "Simulation")
            break
        except Exception as e:
            last_err = e
            Simulation = None
            continue
    if Simulation is None:
        raise ImportError(f"Could not import Simulation: {last_err}")

    Trapping_Vars = None
    for vmod in (
        "trapping_variables",
        "sim.simulation",
        "voltage_interfaceMixin",
        "voltage_fitsMixin",
    ):
        try:
            m = __import__(vmod, fromlist=["Trapping_Vars"])
            Trapping_Vars = getattr(m, "Trapping_Vars")
            break
        except Exception:
            continue
    if Trapping_Vars is None:
        try:
            Trapping_Vars = getattr(sim, "Trapping_Vars")
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "Could not import Trapping_Vars; set PYTHONPATH or adjust app imports."
            ) from e

    return Simulation, Trapping_Vars


def _hashable_cfg(cfg: Dict[str, Any]) -> str:
    """Stable hash key for caching, based on a JSON dump."""
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def _extract_eq_positions(sim, nmf, num_ions):
    import numpy as np

    # 1) Try nmf dict keys
    for key in ("equilibrium_positions", "eq_positions", "equilibrium", "x_eq"):
        if key in nmf:
            arr = np.asarray(nmf[key])
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr

    # 2) Try attributes on sim (dict keyed by num_ions or direct array)
    for attr in (
        "equilibrium_positions",
        "eq_positions",
        "equilibrium",
        "V_min_positions",
        "Vmin_positions",
        "V_minima",
    ):
        if hasattr(sim, attr):
            obj = getattr(sim, attr)
            if isinstance(obj, dict) and num_ions in obj:
                arr = np.asarray(obj[num_ions])
                if arr.ndim == 2 and arr.shape[1] == 3:
                    return arr
            elif isinstance(obj, (list, np.ndarray)):
                arr = np.asarray(obj)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    return arr

    # 3) Last resort: call a method that returns positions
    for meth in (
        "get_equilibrium_positions",
        "find_V_min",
        "find_Vmin",
        "find_V_min_positions",
    ):
        if hasattr(sim, meth):
            try:
                ret = getattr(sim, meth)(num_ions)
                arr = np.asarray(ret)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    return arr
            except Exception:
                pass

    return None


def _scan_equilibrium_candidates(sim, nmf, num_ions):
    import numpy as np

    candidates = []

    # (A) Look inside nmf dict
    for key in (
        "equilibrium_positions",
        "eq_positions",
        "equilibrium",
        "x_eq",
        "V_min_positions",
    ):
        if key in nmf:
            try:
                arr = np.asarray(nmf[key])
                if (
                    arr.ndim == 2
                    and arr.shape[1] == 3
                    and np.issubdtype(arr.dtype, np.number)
                ):
                    candidates.append((f"nmf.{key}", arr))
            except Exception:
                pass

    # (B) Look on sim as attributes
    for attr in (
        "equilibrium_positions",
        "eq_positions",
        "equilibrium",
        "V_min_positions",
        "Vmin_positions",
        "V_minima",
        "x_eq",
    ):
        if hasattr(sim, attr):
            try:
                obj = getattr(sim, attr)
                if isinstance(obj, dict) and num_ions in obj:
                    arr = np.asarray(obj[num_ions])
                    if (
                        arr.ndim == 2
                        and arr.shape[1] == 3
                        and np.issubdtype(arr.dtype, np.number)
                    ):
                        candidates.append((f"sim.{attr}[{num_ions}]", arr))
                else:
                    arr = np.asarray(obj)
                    if (
                        arr.ndim == 2
                        and arr.shape[1] == 3
                        and np.issubdtype(arr.dtype, np.number)
                    ):
                        candidates.append((f"sim.{attr}", arr))
            except Exception:
                pass

    # (C) Try a method
    for meth in (
        "get_equilibrium_positions",
        "find_V_min",
        "find_Vmin",
        "find_V_min_positions",
    ):
        if hasattr(sim, meth):
            try:
                ret = getattr(sim, meth)(num_ions)
                arr = np.asarray(ret)
                if (
                    arr.ndim == 2
                    and arr.shape[1] == 3
                    and np.issubdtype(arr.dtype, np.number)
                ):
                    candidates.append((f"sim.{meth}()", arr))
            except Exception:
                pass

    # Deduplicate by label
    seen = set()
    uniq = []
    for lab, arr in candidates:
        if lab not in seen:
            uniq.append((lab, arr))
            seen.add(lab)
    return uniq


@st.cache_data(show_spinner=False)
def compute_result(cfg_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Core compute: build trap, run stack, collect outputs. Cached by cfg_key."""
    t0 = time.time()
    def _log_step(label: str, start_t: float):
        dt = time.time() - start_t
        print(f"[timing] {label}: {dt:.3f}s")
        return time.time()

    Simulation, Trapping_Vars = _ensure_imports()  # may raise
    t0 = _log_step("imports", t0)

    # Build Trapping_Vars and drives
    tv = Trapping_Vars()
    rf = tv.add_driving(
        "RF", cfg["rf_freq"], 0.0, {"RF1": cfg["rf_amp1"], "RF2": cfg["rf_amp2"]}
    )
    t0 = _log_step("build drives", t0)

    # Apply manual DC electrode offsets on the DC drive first
    ea = tv.Var_dict[tv.dc_key]  # DC drive amplitudes
    if cfg.get("dc_3e_endcaps") or cfg.get("dc_3e_center"):
        tv.add_endcaps_center_3E(cfg["dc_3e_endcaps"], cfg["dc_3e_center"])
    if cfg.get("dc_5e_endcaps") or cfg.get("dc_5e_mid") or cfg.get("dc_5e_center"):
        tv.add_endcaps_mid_center_5E(
            cfg["dc_5e_endcaps"], cfg["dc_5e_mid"], cfg["dc_5e_center"]
        )
    if cfg.get("manual_dc_enabled"):
        for el, V in cfg["manual_dc_offsets"].items():
            try:
                ea.add_amplitude_volt(el, float(V))
            except Exception:
                pass
    if cfg.get("manual_dc_2d_enabled"):
        for el, V in cfg["manual_dc_2d_offsets"].items():
            try:
                ea.add_amplitude_volt(el, float(V))
            except Exception:
                pass
    t0 = _log_step("apply manual DC offsets", t0)

    if cfg.get("apply_twist_endcaps", True):
        tv.apply_dc_twist_endcaps(twist=cfg["twist"], endcaps=float(cfg["endcaps"]))
    t0 = _log_step("apply twist/endcaps", t0)

    if cfg["use_extra"]:
        try:
            extra_map = (
                json.loads(cfg["extra_map_json"]) if cfg["extra_map_json"] else {}
            )
            assert isinstance(extra_map, dict)
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"Extra drive map must be a JSON dict. Parse error: {e}"
            ) from e
        tv.add_driving("ExtraDrive1", cfg["extra_freq"], 0.0, extra_map)
    t0 = _log_step("extra drive", t0)

    # Build Simulation and run base pipeline
    sim = Simulation(cfg["preset"], tv)
    t0 = _log_step("init Simulation", t0)

    # Core pipeline: equilibrium positions, then normal modes/frequencies
    if hasattr(sim, "find_equilib_position_single"):
        sim.find_equilib_position_single(
            num_ions=cfg["num_ions"],
            minimizertype=cfg["minimizer_type"],
        )
        if cfg["num_ions"] != 1:
            sim.find_equilib_position_single(
                num_ions=1,
                minimizertype=cfg["minimizer_type"],
            )
    else:
        raise RuntimeError("Simulation does not have find_equilib_position_single.")
    t0 = _log_step("equilibrium positions", t0)

    if hasattr(sim, "get_static_normal_modes_and_freq"):
        sim.get_static_normal_modes_and_freq(
            cfg["num_ions"], normalize=True, sort_by_freq=True
        )
    else:
        raise RuntimeError("Simulation does not have get_static_normal_modes_and_freq.")
    t0 = _log_step("normal modes/freq", t0)
        
    # --- Static_TotalV plots ---
    plane_cuts_png = None
    along_axes_png = None
    mm_plot_png = None
    if cfg.get("generate_voltage_plots"):
        try:
            fig_plane = sim.plot_total_voltage_plane_cuts(
                n=120, poly_deg=cfg["poly_deg"]
            )
            plane_cuts_png = fig_to_png_bytes(fig_plane)
            fig_axes = sim.plot_total_voltage_along_axes()
            along_axes_png = fig_to_png_bytes(fig_axes)
            # avoid memory leak in Streamlit reruns
            try:
                import matplotlib.pyplot as plt
                plt.close(fig_plane)
                plt.close(fig_axes)
            except Exception:
                pass
        except Exception:
            plane_cuts_png = None
            along_axes_png = None
        t0 = _log_step("static voltage plots", t0)

    if cfg.get("generate_mm_plot"):
        try:
            fig_mm = sim.plot_total_MM_magnitude()
            mm_plot_png = fig_to_png_bytes(fig_mm)
            try:
                import matplotlib.pyplot as plt
                plt.close(fig_mm)
            except Exception:
                pass
        except Exception:
            mm_plot_png = None
        t0 = _log_step("micromotion plot", t0)


    # # Frequencies & modes
    # nmf = sim.normal_modes_and_frequencies[cfg["num_ions"]]
    # freqs = np.asarray(nmf.get("frequencies_Hz"), dtype=float)
    # modes = np.asarray(nmf.get("modes"))  # (3N, 3N) with modes as columns

    # Frequencies & modes in the PRINCIPAL basis (columns are modes)
    # Ensure principal objects exist (get_static_normal_modes_and_freq already populates them)
    try:
        ppack = sim.principal_dir_normalmodes_andfrequencies.get(cfg["num_ions"])
    except Exception:
        ppack = None
    if ppack is None:
        # Be explicit in case this run path didn’t build them yet
        sim.compute_principal_directions_from_one_ion()
        sim.populate_normalmodes_in_prinipledir_freq_labels()
        ppack = sim.principal_dir_normalmodes_andfrequencies.get(cfg["num_ions"])
    t0 = _log_step("principal directions", t0)

    freqs = np.asarray(ppack.get("frequencies_Hz"), dtype=float)
    modes = np.asarray(ppack.get("modes"), dtype=float)  # principal basis (3N×3N)
    dir_alignment = np.asarray(ppack.get("dir_alignment"))  # (3N,) 0/1/2 per mode

    # Also surface the 3 principal directions in lab coords
    principal_dirs = np.asarray(
        sim.principal_dirs, dtype=float
    )  # shape (3,3); rows dir_0..2 => [x,y,z]

    # Single-ion secular frequencies along the principal directions (Hz)
    # (dir_0 is the lowest-frequency principal axis)
    secular_freqs_Hz = None
    try:
        p1 = sim.principal_dir_normalmodes_andfrequencies.get(1)
        if p1 is None:
            # Ensure 1-ion data exists
            sim.get_static_normal_modes_and_freq(1, normalize=True, sort_by_freq=True)
            sim.populate_normalmodes_in_prinipledir_freq_labels()
            p1 = sim.principal_dir_normalmodes_andfrequencies.get(1)
        if p1 is not None:
            secular_freqs_Hz = np.asarray(p1.get("frequencies_Hz"), dtype=float)
    except Exception:
        secular_freqs_Hz = None
    t0 = _log_step("1-ion secular freqs", t0)

    # Equilibrium positions
    eq_positions = _eq_from_ion_equilibrium_positions(sim, cfg["num_ions"])
    eq_metadata = _eq_metadata_from_sim(sim, cfg["num_ions"], cfg["minimizer_type"])

    # Resonances
    out_res = None
    if cfg.get("compute_resonant_couplings"):
        drives_arg = None  # let the sim discover all non-DC drives
        try:
            out_res = sim.collect_resonant_couplings(
                num_ions=cfg["num_ions"],
                tol_Hz=cfg["tol_Hz"],
                orders=tuple(cfg["orders"]),
                drives=drives_arg,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "collect_resonant_couplings failed. Ensure StaticCoupolingMixin with this method is mixed into Simulation."
            ) from e
        t0 = _log_step("resonant couplings", t0)

    return {
        "frequencies_Hz": freqs,
        "modes": modes,
        "dir_alignment": dir_alignment,  # optional but useful in UI
        "eq_positions": eq_positions,
        "eq_metadata": eq_metadata,
        "resonances": out_res,
        "principal_dirs": principal_dirs,  # 3×3, rows: dir_0..2 in lab (x,y,z)
        "secular_frequencies_Hz": secular_freqs_Hz,  # (3,) for 1-ion if available
        # new: Static_TotalV plots
        "static_totalV_plane_cuts_png": plane_cuts_png,
        "static_totalV_along_axes_png": along_axes_png,
        "generate_voltage_plots": bool(cfg.get("generate_voltage_plots")),
        "mm_plot_png": mm_plot_png,
        "generate_mm_plot": bool(cfg.get("generate_mm_plot")),
    
    }


# ------------------------------------------------------------
# Optional: warn if inputs changed since last compute
# ------------------------------------------------------------
pending_cfg = {
    "preset": preset,
    "num_ions": int(num_ions),
    "minimizer_type": str(minimizer_type),
    "poly_deg": int(poly_deg),
    "generate_voltage_plots": bool(generate_voltage_plots),
    "compute_resonant_couplings": bool(compute_resonant_couplings),
    "generate_mm_plot": bool(generate_mm_plot),
    "rf_freq": float(rf_freq),
    "rf_amp1": float(rf_amp1),
    "rf_amp2": float(rf_amp2),
    "use_extra": bool(use_extra),
    "extra_freq": float(extra_freq),
    "extra_map_json": extra_map_json,
    "twist": float(twist),
    "endcaps": float(endcaps),
    "apply_twist_endcaps": bool(apply_twist_endcaps),
    "tol_Hz": float(tol_Hz),
    "orders": list(map(int, orders_pick)),
    "manual_dc_enabled": bool(manual_dc_enabled),
    "manual_dc_offsets": (
        {k: float(v) for k, v in st.session_state["manual_dc"].items()}
        if manual_dc_enabled
        else {}
    ),
    "manual_dc_2d_enabled": bool(manual_dc_2d_enabled),
    "manual_dc_2d_offsets": (
        {k: float(v) for k, v in st.session_state["manual_dc_2d"].items()}
        if manual_dc_2d_enabled
        else {}
    ),
    "dc_3e_endcaps": float(dc_3e_endcaps),
    "dc_3e_center": float(dc_3e_center),
    "dc_5e_endcaps": float(dc_5e_endcaps),
    "dc_5e_mid": float(dc_5e_mid),
    "dc_5e_center": float(dc_5e_center),
}

# ------------------------------------------------------------
# Run compute only when requested; otherwise render from cache
# ------------------------------------------------------------
if run_btn:
    cfg = dict(pending_cfg)  # snapshot
    cfg_key = _hashable_cfg(cfg)
    if cfg.get("minimizer_type") == "Quartic2D_101":
        cfg_key = f"{cfg_key}:{time.time_ns()}"
    try:
        t0 = time.time()
        res = compute_result(cfg_key, cfg)
        dt = time.time() - t0
    except Exception as e:
        st.error(f"❌ Compute failed: {e}")
        res = None
    else:
        st.session_state["res"] = res
        st.session_state["cfg"] = cfg
        st.success(f"✅ Done in {dt:.3f} s")

# Always render from cached result (if any)
res = st.session_state.get("res")
cfg_cached = st.session_state.get("cfg")

if cfg_cached and _hashable_cfg(pending_cfg) != _hashable_cfg(cfg_cached):
    st.warning("Inputs changed since last compute — press **Compute** to update.")

# Small box with principal directions (lab coords)
if res is not None:
    with st.container():
        st.subheader("Principal directions (lab basis)")
        P = res.get(
            "principal_dirs", None
        )  # shape (3,3); rows = dir_0..2, cols = [x,y,z]
        if P is not None:
            dfP = pd.DataFrame(np.asarray(P, dtype=float), columns=["x", "y", "z"])
            dfP.index = ["dir_0 (lowest f)", "dir_1", "dir_2"]

            sec = res.get("secular_frequencies_Hz", None)
            if sec is not None:
                sec = np.asarray(sec, dtype=float).reshape(-1)
                if sec.size >= 3:
                    dfP["f (MHz)"] = sec[:3] / 1e6

            st.dataframe(
                dfP.style.format("{:.6f}"),
                use_container_width=False,
                height=130,
            )
        else:
            st.caption("Principal directions not available.")

# Static_TotalV plots
if res is not None and res.get("generate_voltage_plots"):
    png_plane = res.get("static_totalV_plane_cuts_png")
    if png_plane:
        st.subheader("Static_TotalV plane cuts")
        st.image(png_plane, use_column_width=True)
    png_axes = res.get("static_totalV_along_axes_png")
    if png_axes:
        st.subheader("Static_TotalV along axes")
        st.image(png_axes, use_column_width=True)

if res is not None and res.get("generate_mm_plot"):
    png_mm = res.get("mm_plot_png")
    if png_mm:
        st.subheader("Micromotion magnitude")
        st.image(png_mm, use_column_width=True)



if res is not None:
    # --------------------------------------------------------
    # Layout: two columns for quick glance
    # --------------------------------------------------------
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Equilibrium positions (µm)")
        arr_to_show = res.get("eq_positions")

        if arr_to_show is None:
            st.info(
                f"No entry in sim.ion_equilibrium_positions for {pending_cfg['num_ions']} ions. "
                "Make sure your pipeline populates it (your _smoke_test_new_stack already should)."
            )
        else:
            df_eq = (
                pd.DataFrame(np.asarray(arr_to_show), columns=["x", "y", "z"])
                .reset_index()
                .rename(columns={"index": "ion"})
            )
            # Show in µm if values look like meters
            scale = (
                1e6 if np.nanmax(np.abs(df_eq[["x", "y", "z"]].values)) < 1e-2 else 1.0
            )
            if scale != 1.0:
                df_eq[["x", "y", "z"]] = df_eq[["x", "y", "z"]] * scale
            st.caption("source: sim.ion_equilibrium_positions[num_ions]")
            st.dataframe(df_eq, use_container_width=True)

        if show_equilibrium_viewer:
            st.subheader("Equilibrium viewer")
            eq_metadata = res.get("eq_metadata")
            if arr_to_show is None:
                st.caption("No equilibrium positions are available for plotting.")
            elif not isinstance(eq_metadata, dict):
                st.caption("No equilibrium metadata is available for this run.")
            elif not _HAS_MPL or build_equilibrium_figure is None:
                st.caption("Matplotlib equilibrium viewer is unavailable in this environment.")
            else:
                fig = None
                try:
                    fig = build_equilibrium_figure(
                        arr_to_show,
                        seed_positions=eq_metadata.get("seed_positions"),
                        plane_normal=eq_metadata.get("plane_normal"),
                        title="Equilibrium positions",
                    )
                except Exception:
                    fig = None

                if fig is not None:
                    st.pyplot(fig)
                    try:
                        import matplotlib.pyplot as plt

                        plt.close(fig)
                    except Exception:
                        pass
                else:
                    st.caption("Equilibrium viewer data could not be rendered.")

                summary_lines = []
                if build_equilibrium_summary_lines is not None:
                    try:
                        summary_lines = build_equilibrium_summary_lines(eq_metadata)
                    except Exception:
                        summary_lines = []
                if summary_lines:
                    st.text("\n".join(summary_lines))

        st.subheader("Frequencies (Hz)")
        df_f = pd.DataFrame(
            {
                "mode": np.arange(len(res["frequencies_Hz"])),
                "f_Hz": res["frequencies_Hz"],
            }
        )

        # attach dir label per mode (fallback to "n/a" if alignment missing)
        align = res.get("dir_alignment")
        if isinstance(align, (list, np.ndarray)) and len(align) >= len(df_f):
            df_f["dir"] = [int(a) for a in np.asarray(align)[: len(df_f)]]
        else:
            df_f["dir"] = [-1] * len(df_f)
        df_f["dir_label"] = (
            df_f["dir"].map({0: "dir_0", 1: "dir_1", 2: "dir_2"}).fillna("n/a")
        )

        st.dataframe(df_f[["mode", "f_Hz", "dir_label"]], use_container_width=True)

        # color bars by principal direction using your palette
        chart = (
            alt.Chart(df_f)
            .mark_bar()
            .encode(
                x=alt.X("mode:O", title="mode"),
                y=alt.Y("f_Hz:Q", title="Frequency (Hz)"),
                color=alt.Color(
                    "dir_label:N",
                    scale=alt.Scale(
                        domain=["dir_0", "dir_1", "dir_2"],
                        range=[DIR_COLORS[0], DIR_COLORS[1], DIR_COLORS[2]],
                    ),
                    legend=alt.Legend(title="principal dir"),
                ),
                tooltip=[
                    alt.Tooltip("mode:O"),
                    alt.Tooltip("f_Hz:Q", format=".2f"),
                    alt.Tooltip("dir_label:N", title="dir"),
                ],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, use_container_width=True)

    with c2:
        st.subheader("Normal modes |principal basis| (columns are modes)")
        modes = res["modes"]
        if modes is None or modes.size == 0:
            st.info("Mode matrix not available.")
        else:
            abs_modes = np.abs(modes)
            ncoords = abs_modes.shape[0]
            labels = [f"q{idx+1}" for idx in range(ncoords)]
            df_modes = pd.DataFrame(abs_modes, index=labels)

            st.dataframe(
                df_modes.style.format("{:.3e}"),
                use_container_width=True,
                height=360,
            )

        # --- Per-mode participation bar plot ---
        st.subheader("Mode participation (per ion)")
        if modes is not None and modes.size != 0:
            import matplotlib.pyplot as plt

            freqs = np.asarray(res.get("frequencies_Hz"), dtype=float)
            M = modes.shape[1]  # 3N
            N = modes.shape[0] // 3

            # Mode selector (aligns with freqs and modes columns)
            mode_labels = [
                f"{i}: {freqs[i]:.2f} Hz" if i < len(freqs) else f"{i}"
                for i in range(M)
            ]
            mode_idx = st.selectbox(
                "Select mode",
                options=list(range(M)),
                format_func=lambda i: mode_labels[i],
                index=0,
                key="mode_select",
            )
            plot_abs = st.checkbox("Plot absolute participation |·|", value=True)

            # Pull the selected eigenvector (columns are modes)
            v = np.asarray(modes[:, mode_idx]).reshape(3 * N)
            # Build per-ion xyz arrays
            d0_part = v[0::3]
            d1_part = v[1::3]
            d2_part = v[2::3]

            if plot_abs:
                d0_show, d1_show, d2_show = (
                    np.abs(d0_part),
                    np.abs(d1_part),
                    np.abs(d2_part),
                )
                y_label = "Participation (|component|)"
            else:
                d0_show, d1_show, d2_show = d0_part, d1_part, d2_part
                y_label = "Participation (component)"

            # Grouped bars positions (unchanged geometry)
            group_gap = 0.6
            base = np.arange(N) * (3.0 + group_gap)
            pos_d0 = base + 0.0
            pos_d1 = base + 1.0
            pos_d2 = base + 2.0
            width = 1.0

            # X tick labels: ion index and (x,y,z) if available
            eq = res.get("eq_positions")
            if eq is not None:
                eq = np.asarray(eq, dtype=float)
                scale = 1e6 if np.nanmax(np.abs(eq)) < 1e-2 else 1.0
                unit = "µm" if scale == 1e6 else "arb"
                disp = eq * scale
                tick_labels = [
                    f"{i+1}\n({disp[i,0]:.2f}, {disp[i,1]:.2f}, {disp[i,2]:.2f}) {unit}"
                    for i in range(N)
                ]
            else:
                tick_labels = [f"{i+1}" for i in range(N)]

            fig, ax = plt.subplots(figsize=(min(10, max(6, N * 0.6)), 4.8))

            # Colors for dir_0/1/2 (re-use your palette)
            col_d0 = "#1f77b4"  # dir_0
            col_d1 = "#ff7f0e"  # dir_1
            col_d2 = "#2ca02c"  # dir_2

            ax.bar(pos_d0, d0_show, width=width, label="dir_0", color=col_d0)
            ax.bar(pos_d1, d1_show, width=width, label="dir_1", color=col_d1)
            ax.bar(pos_d2, d2_show, width=width, label="dir_2", color=col_d2)

            # Title (with alignment info if available)
            f_str = f"{freqs[mode_idx]:.2f} Hz" if mode_idx < len(freqs) else "n/a"
            align = res.get("dir_alignment", None)
            if isinstance(align, (list, np.ndarray)) and mode_idx < len(align):
                ax.set_title(
                    f"Mode {mode_idx} — f = {f_str} — aligns with dir_{int(align[mode_idx])}"
                )
            else:
                ax.set_title(f"Mode {mode_idx} — f = {f_str}")

            ax.set_ylabel(y_label)
            ax.set_xticks(base + 1.0)
            ax.set_xticklabels(tick_labels, rotation=0, ha="center")
            ax.grid(axis="y", linestyle=":", alpha=0.4)

            if not plot_abs:
                ymax = np.nanmax(np.abs([d0_show, d1_show, d2_show]))
                ax.set_ylim(-1.1 * ymax, 1.1 * ymax)

            ax.legend(ncols=3, loc="upper right", frameon=False)
            st.pyplot(fig, use_container_width=True)

        else:
            st.info("Mode matrix not available, so participation plot can’t be drawn.")

        st.subheader("Resonant couplings")
        out_res = res["resonances"]
        rows: List[Dict[str, Any]] = []
        try:
            R = out_res["resonances"]
            for order in (2, 3, 4):
                for item in R.get(order, []):
                    row = {
                        "order": order,
                        "modes": tuple(item.get("modes", [])),
                        "target_Hz": item.get("target_Hz"),
                        "detune_Hz": item.get("delta_Hz"),
                    }
                    if order == 2:
                        row["g0_Hz"] = item.get("g0_Hz") or item.get("g0_Hz_by_drive")
                        if "drive_resonances" in item:
                            row["drives"] = [
                                getattr(d.get("drive"), "label", str(d.get("drive")))
                                for d in item["drive_resonances"]
                            ]
                    elif order == 3:
                        row["g3_Hz"] = item.get("g3_Hz")
                    elif order == 4:
                        row["g4_Hz"] = item.get("g4_Hz")
                    rows.append(row)
        except Exception:
            st.info("Resonance output schema not recognized; showing raw JSON below.")
            rows = []

        if rows:
            df_rows = pd.DataFrame(rows)
            st.dataframe(
                df_rows.sort_values(["order", "detune_Hz"], ascending=[True, True]),
                use_container_width=True,
            )
        else:
            st.write("(No flattened rows to display.)")

    # ────────────────────────────────────────────────────────────────────────────
    # On-demand F110 max matrix (bounds ±1 V), separate Run button
    # ────────────────────────────────────────────────────────────────────────────
    with st.expander("F110 Max Coupling Matrix (bounds ±1 V)"):
        st.write(
            "Click **Run** to compute the per-pair max driven g₀ (Hz) within ±1 V "
            "on each symmetric input (DC1=DC10, DC2=DC9, …). This may take longer."
        )
        cA, cB = st.columns([1, 1])
        run_max = cA.button("Run F110 Max", key="btn_run_f110_max")
        clear_max = cB.button("Clear", key="btn_clear_f110_max")

        if clear_max:
            st.session_state.pop("f110_max_matrix", None)
            st.session_state.pop("f110_max_meta", None)

        if run_max:
            # Rebuild a Trapping_Vars that matches the current cached config
            cfg_src = cfg_cached if cfg_cached is not None else pending_cfg
            try:
                SimulationCls, Trapping_VarsCls = _ensure_imports(
                    cfg_src["repo_path"], cfg_src["add_repo_to_sys_path"]
                )
                tv2 = Trapping_VarsCls()
                # RF
                tv2.add_driving(
                    "RF",
                    cfg_src["rf_freq"],
                    0.0,
                    {"RF1": cfg_src["rf_amp1"], "RF2": cfg_src["rf_amp2"]},
                )

                # Manual DC → add to DC drive before twist/endcaps
                if cfg_src.get("manual_dc_enabled") and cfg_src.get(
                    "manual_dc_offsets"
                ):
                    ea2 = tv2.Var_dict[tv2.dc_key]  # DC drive amplitudes
                    for el, V in cfg_src["manual_dc_offsets"].items():
                        try:
                            ea2.add_amplitude_volt(el, float(V))
                        except Exception:
                            pass
                # Geometry knobs that bit us before
                tv2.apply_dc_twist_endcaps(
                    twist=cfg_src["twist"], endcaps=float(cfg_src["endcaps"])
                )
                # Optional extra drive (not required, but keeps parity with the main sim)
                if cfg_src["use_extra"]:
                    try:
                        extra_map = (
                            json.loads(cfg_src["extra_map_json"])
                            if cfg_src["extra_map_json"]
                            else {}
                        )
                    except Exception:
                        extra_map = {}
                    tv2.add_driving(
                        "ExtraDrive1", cfg_src["extra_freq"], 0.0, extra_map
                    )

                with st.spinner("Computing F110 max matrix…"):
                    Mmax = compute_f110_max_matrix(
                        tv=tv2,
                        num_ions=int(cfg_src["num_ions"]),
                        preset=cfg_src["preset"],
                        point=[0.0, 0.0, 0.0, 0.0, 0.0],  # evaluate at origin
                        bounds=[(-1.0, 1.0)] * 5,  # ±1 V on each symmetric DOF
                    )
                st.session_state["f110_max_matrix"] = Mmax
                st.session_state["f110_max_meta"] = {
                    "N": int(cfg_src["num_ions"]),
                    "preset": cfg_src["preset"],
                    "bounds": "±1 V",
                }
                st.success("F110 max matrix computed.")
            except Exception as e:
                st.error(f"F110 max computation failed: {e}")

        if "f110_max_matrix" in st.session_state:
            Mmax = st.session_state["f110_max_matrix"]
            meta = st.session_state.get("f110_max_meta", {})

            st.caption(
                f"Upper-triangular matrix of per-pair maxima (Hz). "
                f"N={meta.get('N','?')}, preset={meta.get('preset','?')}, bounds={meta.get('bounds','?')}."
            )

            df = pd.DataFrame(Mmax)
            vmax = float(np.nanmax(Mmax)) if np.size(Mmax) else 0.0
            vmin_visible = 1.0  # anything ≥1 Hz should show noticeably blue

            # def _cell_color_log(val, _vmin=vmin_visible, _vmax=vmax):
            #     try:
            #         x = float(val)
            #     except Exception:
            #         return "background-color: rgb(0,0,0)"
            #     if not np.isfinite(x) or x <= 0.0:
            #         return "background-color: rgb(0,0,0)"
            #     if _vmax <= _vmin:
            #         t = 1.0
            #     else:
            #         # log10 scaling so values near 1 Hz are visible even if max ≫ 1
            #         t = (np.log10(x) - np.log10(_vmin)) / (
            #             np.log10(_vmax) - np.log10(_vmin)
            #         )
            #         t = min(1.0, max(0.0, t))
            #     b = int(round(255 * t))  # black → blue
            #     return f"background-color: rgb(0,0,{b})"

            def _cell_color_log(val, _vmin=vmin_visible, _vmax=vmax):
                try:
                    x = float(val)
                except Exception:
                    # neutral dark cell with light text
                    return "background-color: #0e0f13; color: #eaeaea"
                if not np.isfinite(x) or x <= 0.0:
                    return "background-color: #0e0f13; color: #eaeaea"
                if _vmax <= _vmin:
                    t = 1.0
                else:
                    # log scale for visibility across decades
                    t = (np.log10(x) - np.log10(_vmin)) / (
                        np.log10(_vmax) - np.log10(_vmin)
                    )
                    t = min(1.0, max(0.0, t))
                b = int(round(255 * t))  # black → blue
                return f"background-color: rgb(0,0,{b}); color: #eaeaea"

            styled = df.style.format("{:.3e}").applymap(_cell_color_log)
            st.dataframe(styled, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────────
    # F110 Targeted Coupling Solver (enter desired upper-triangle, hit Compute)
    # ────────────────────────────────────────────────────────────────────────────
    with st.expander("F110 Targeted Coupling Solver (enter desired g₀ targets, Hz)"):
        st.write(
            "Enter a **K×K** matrix of desired g₀ couplings (Hz). "
            "Only the **upper triangle** (i<j) is used; diagonal and lower triangle are ignored. "
            "Then click **Compute** to solve for the 5 symmetric DC inputs."
        )

        # Determine dimension K = 3N
        K = 3 * int(num_ions)

        # Keep an editable DataFrame in session_state so it doesn’t reset on rerun
        key_df = f"f110_target_matrix_K{K}"
        if key_df not in st.session_state:
            st.session_state[key_df] = pd.DataFrame(
                np.zeros((K, K), dtype=float),
                index=[f"i={i}" for i in range(K)],  # row labels
                columns=[f"j={j}" for j in range(K)],  # col labels
            )

        # Editor
        df_target = st.data_editor(
            st.session_state[key_df],
            use_container_width=True,
            num_rows="dynamic",
            key=f"editor_{key_df}",
            column_config=None,
        )

        c1, c2, c3 = st.columns([1, 1, 2])
        compute_btn = c1.button("Compute inputs", key="btn_f110_targets_compute")
        clear_btn = c2.button("Clear matrix", key="btn_f110_targets_clear")
        note = c3.caption("Tip: fill only cells with i<j. Others will be ignored.")

        if clear_btn:
            st.session_state[key_df] = pd.DataFrame(
                np.zeros((K, K), dtype=float),
                index=[f"i={i}" for i in range(K)],
                columns=[f"j={j}" for j in range(K)],
            )
            df_target = st.session_state[key_df]

        if compute_btn:
            # Build target list from upper triangle
            targets = []
            for i in range(K):
                for j in range(i + 1, K):
                    try:
                        val = float(df_target.iat[i, j])
                    except Exception:
                        val = 0.0
                    if val != 0.0:  # include only nonzero targets
                        targets.append(((i, j), val))

            if not targets:
                st.warning(
                    "No upper-triangle targets entered. Fill some (i<j) entries and try again."
                )
            else:
                # Rebuild a Trapping_Vars matching the current cached config (parity with other sections)
                cfg_src = cfg_cached if cfg_cached is not None else pending_cfg
                try:
                    SimulationCls, Trapping_VarsCls = _ensure_imports(
                        cfg_src["repo_path"], cfg_src["add_repo_to_sys_path"]
                    )
                    tv2 = Trapping_VarsCls()
                    # RF
                    tv2.add_driving(
                        "RF",
                        cfg_src["rf_freq"],
                        0.0,
                        {"RF1": cfg_src["rf_amp1"], "RF2": cfg_src["rf_amp2"]},
                    )

                    # Manual DC → add to DC drive before twist/endcaps
                    if cfg_src.get("manual_dc_enabled") and cfg_src.get(
                        "manual_dc_offsets"
                    ):
                        ea2 = tv2.Var_dict[tv2.dc_key]  # DC drive amplitudes
                        for el, V in cfg_src["manual_dc_offsets"].items():
                            try:
                                ea2.add_amplitude_volt(el, float(V))
                            except Exception:
                                pass

                    # Geometry knobs (twist / endcaps) – keep parity with main sim
                    tv2.apply_dc_twist_endcaps(
                        twist=cfg_src["twist"], endcaps=float(cfg_src["endcaps"])
                    )
                    # Optional extra drive map if you’re carrying one around; safe to skip
                    if cfg_src.get("use_extra"):
                        try:
                            extra_map = json.loads(
                                cfg_src.get("extra_map_json") or "{}"
                            )
                        except Exception:
                            extra_map = {}
                        tv2.add_driving(
                            "ExtraDrive1", cfg_src["extra_freq"], 0.0, extra_map
                        )

                    with st.spinner("Solving F110 for requested targets…"):
                        result = solve_F110_for_targets(
                            mode_pair_targets=targets,  # [((i,j), Hz), ...]
                            num_ions=int(cfg_src["num_ions"]),
                            constant_trappingvars=tv2,
                            point=[
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                            ],  # solve at origin (linear map)
                            simulation_preset=cfg_src["preset"],
                            minimize_rest=False,  # optional
                            reg_l2=0.0,  # no regularization by default
                        )

                    # Show solution
                    st.success(
                        "Solved inputs (symmetric DC DOFs: DC1=10, DC2=9, DC3=8, DC4=7, DC5=6)"
                    )
                    u = result["u_star"]
                    cols = st.columns(5)
                    for k, lab in enumerate(
                        ["(1=10)", "(2=9)", "(3=8)", "(4=7)", "(5=6)"]
                    ):
                        cols[k].metric(f"u[{k+1}] {lab}", f"{u[k]:.6f} V")

                    # Achieved vs target table
                    ach = result["achieved_Hz"]
                    err = result["achieved_err_Hz"]
                    rows = []
                    for (i, j), tgt in targets:
                        rows.append(
                            {
                                "pair (i,j)": f"({i},{j})",
                                "target (Hz)": tgt,
                                "achieved (Hz)": ach[(i, j)],
                                "error (Hz)": err[(i, j)],
                            }
                        )
                    st.write("Achieved vs target:")
                    st.dataframe(
                        pd.DataFrame(rows).style.format(
                            {
                                "target (Hz)": "{:.6e}",
                                "achieved (Hz)": "{:.6e}",
                                "error (Hz)": "{:.6e}",
                            }
                        ),
                        use_container_width=True,
                    )

                    # Diagnostics
                    st.caption(
                        f"rank={result['rank']}, nullspace_dim={result['nullspace_dim']}, "
                        f"pred_rest_norm={result['pred_rest_norm_Hz']:.3e} Hz"
                    )

                except Exception as e:
                    st.error(f"F110 target solve failed: {e}")

    # ────────────────────────────────────────────────────────────────────────────
    # F310 Max Coupling Tensor (bounds ±1 V) — mirrors F110 max matrix
    # ────────────────────────────────────────────────────────────────────────────
    with st.expander("F310 Max Coupling Tensor (bounds ±1 V)"):
        st.write(
            "Click **Run** to compute the per-(a<b, c) max |g₀| (Hz) within ±1 V "
            "on each symmetric input (DC1=DC10, DC2=DC9, …). Results persist until **Clear**."
        )
        cA, cB = st.columns([1, 1])
        run_f310_max = cA.button("Run F310 Max", key="btn_run_f310_max")
        clear_f310_max = cB.button("Clear", key="btn_clear_f310_max")

        if clear_f310_max:
            st.session_state.pop("f310_max_tensor", None)
            st.session_state.pop("f310_max_meta", None)

        if run_f310_max:
            # Build a fresh Trapping_Vars like the F110 section
            cfg_src = cfg_cached if cfg_cached is not None else pending_cfg
            try:
                SimulationCls, Trapping_VarsCls = _ensure_imports(
                    cfg_src["repo_path"], cfg_src["add_repo_to_sys_path"]
                )
                tv2 = Trapping_VarsCls()
                # RF drive
                tv2.add_driving(
                    "RF",
                    cfg_src["rf_freq"],
                    0.0,
                    {"RF1": cfg_src["rf_amp1"], "RF2": cfg_src["rf_amp2"]},
                )
                # Manual DC offsets (before twist/endcaps)
                if cfg_src.get("manual_dc_enabled") and cfg_src.get(
                    "manual_dc_offsets"
                ):
                    ea2 = tv2.Var_dict[tv2.dc_key]
                    for el, V in cfg_src["manual_dc_offsets"].items():
                        try:
                            ea2.add_amplitude_volt(el, float(V))
                        except Exception:
                            pass
                # Geometry knobs
                tv2.apply_dc_twist_endcaps(
                    twist=cfg_src["twist"], endcaps=float(cfg_src["endcaps"])
                )
                # Optional extra drive (parity with main sim)
                if cfg_src.get("use_extra"):
                    try:
                        extra_map = json.loads(cfg_src.get("extra_map_json") or "{}")
                    except Exception:
                        extra_map = {}
                    tv2.add_driving(
                        "ExtraDrive1", cfg_src["extra_freq"], 0.0, extra_map
                    )

                with st.spinner("Computing F310 max tensor…"):
                    f310 = find_max_coupling_tensor_F310(
                        num_ions=int(cfg_src["num_ions"]),
                        constant_trappingvars=tv2,
                        bounds5=[(-1.0, 1.0)] * 5,
                        point=[0.0] * 5,
                        simulation_preset=cfg_src["preset"],
                        return_argmax=False,
                    )
                # Enforce full permutation symmetry so diagonal entries (a,a,c) are filled too
                st.session_state["f310_max_tensor"] = _symmetrize_rank3_abs(
                    f310["Gmax_Hz"]
                )  # (K×K×K)
                st.session_state["f310_max_meta"] = {
                    "N": int(cfg_src["num_ions"]),
                    "preset": cfg_src["preset"],
                    "bounds": "±1 V",
                    "K": int(f310["K"]),
                }
                st.success("F310 max tensor computed.")
            except Exception as e:
                st.error(f"F310 max computation failed: {e}")

        # Display (slice by c)
        if "f310_max_tensor" in st.session_state:
            Gmax = st.session_state["f310_max_tensor"]
            meta = st.session_state.get("f310_max_meta", {})
            K = int(meta.get("K", Gmax.shape[0]))
            st.caption(
                f"Tensor slices (Hz). Showing Gmax[:, :, c]. "
                f"N={meta.get('N','?')}, preset={meta.get('preset','?')}, bounds={meta.get('bounds','?')}."
            )
            c_idx = st.selectbox(
                "Select c-index (mode)",
                options=list(range(K)),
                index=0,
                key="f310_max_cidx",
            )
            slice_mat = np.asarray(Gmax)[:, :, c_idx]

            df_slice = pd.DataFrame(slice_mat)

            def _style_upper_log(df: pd.DataFrame) -> pd.DataFrame:
                """
                Color ONLY the upper triangle **including the diagonal** (i<=j).
                Log-scale is computed from strictly-positive values on that region.
                Zeros / non-positives / NaNs are treated as 'no signal' (dark).
                """
                arr = df.to_numpy(copy=False)
                n = arr.shape[0]
                css = np.full(arr.shape, "", dtype=object)

                # indices of the upper triangle INCLUDING diagonal
                iu = np.triu_indices(n, k=0)

                # gather strictly positive values on upper+diag for scaling
                upper_vals = arr[iu]
                pos = upper_vals[(upper_vals > 0) & np.isfinite(upper_vals)]
                if pos.size == 0:
                    return pd.DataFrame(css, index=df.index, columns=df.columns)

                vmin = float(pos.min())
                vmax = float(pos.max())
                log_min = math.log10(vmin)
                log_max = math.log10(vmax) if vmax > vmin else log_min + 1e-9

                # build CSS only for upper+diag
                for i, j in zip(*iu):
                    x = arr[i, j]
                    if not np.isfinite(x) or x <= 0.0:
                        css[i, j] = "background-color: #0e0f13; color: #eaeaea"
                    else:
                        t = (math.log10(x) - log_min) / (log_max - log_min)
                        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                        b = int(round(255 * t))  # blue intensity
                        css[i, j] = f"background-color: rgb(0,0,{b}); color: #eaeaea"

                # lower triangle stays unstyled
                return pd.DataFrame(css, index=df.index, columns=df.columns)

            styled = df_slice.style.format("{:.3e}").apply(_style_upper_log, axis=None)
            st.dataframe(styled, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────────
    # F310 Targeted Coupling Solver (3-mode) — mirrors F110 solver with c-slice editor
    # ────────────────────────────────────────────────────────────────────────────
    with st.expander("F310 Targeted Coupling Solver (enter desired g₀ targets) Hz)"):
        st.write(
            "Pick a **c-index** (mode) and edit the **K×K** table for that slice. "
            "Only the **upper triangle** (i<j) is used. Click **Compute** to solve for the 5 symmetric inputs."
        )

        # Dimension K = 3N (consistent with the rest of the app)
        K = 3 * int(num_ions)
        c_idx = st.selectbox(
            "Select c-index (mode for the slice)",
            options=list(range(K)),
            index=0,
            key="f310_target_cidx",
        )

        # Keep an editable DataFrame per (K, c) so it persists across reruns
        key_df = f"f310_target_slice_c{c_idx}_K{K}"
        if key_df not in st.session_state:
            st.session_state[key_df] = pd.DataFrame(
                np.zeros((K, K), dtype=float),
                index=[f"i={i}" for i in range(K)],
                columns=[f"j={j}" for j in range(K)],
            )

        df_target_slice = st.data_editor(
            st.session_state[key_df],
            use_container_width=True,
            num_rows="dynamic",
            key=f"editor_{key_df}",
        )

        c1, c2, c3 = st.columns([1, 1, 2])
        compute_btn = c1.button("Compute inputs (F310)", key="btn_f310_targets_compute")
        clear_btn = c2.button("Clear slice", key="btn_f310_targets_clear")
        c3.caption(
            "Tip: fill only cells with i<j. Others are ignored. Leave 0.0 to skip a pair."
        )

        if clear_btn:
            st.session_state[key_df] = pd.DataFrame(
                np.zeros((K, K), dtype=float),
                index=[f"i={i}" for i in range(K)],
                columns=[f"j={j}" for j in range(K)],
            )
            df_target_slice = st.session_state[key_df]

        if compute_btn:
            # Build a (K×K×K) target tensor with NaN everywhere except this c-slice (upper triangle)
            target = np.full((K, K, K), np.nan, dtype=float)
            nonempty = []
            for i in range(K):
                for j in range(i + 1, K):
                    try:
                        val = float(df_target_slice.iat[i, j])
                    except Exception:
                        val = 0.0
                    if (
                        val != 0.0
                    ):  # mirror F110’s convention: only constrain nonzero entries
                        target[i, j, c_idx] = val
                        nonempty.append((i, j, val))

            if not nonempty:
                st.warning(
                    "No upper-triangle targets entered for this c-slice. Fill some (i<j) entries and try again."
                )
            else:
                # Rebuild a fresh Trapping_Vars like the F110 solve section
                cfg_src = cfg_cached if cfg_cached is not None else pending_cfg
                try:
                    SimulationCls, Trapping_VarsCls = _ensure_imports(
                        cfg_src["repo_path"], cfg_src["add_repo_to_sys_path"]
                    )
                    tv2 = Trapping_VarsCls()
                    # RF
                    tv2.add_driving(
                        "RF",
                        cfg_src["rf_freq"],
                        0.0,
                        {"RF1": cfg_src["rf_amp1"], "RF2": cfg_src["rf_amp2"]},
                    )
                    # Manual DC → DC drive before twist/endcaps
                    if cfg_src.get("manual_dc_enabled") and cfg_src.get(
                        "manual_dc_offsets"
                    ):
                        ea2 = tv2.Var_dict[tv2.dc_key]
                        for el, V in cfg_src["manual_dc_offsets"].items():
                            try:
                                ea2.add_amplitude_volt(el, float(V))
                            except Exception:
                                pass
                    # Geometry knobs (parity with main sim)
                    tv2.apply_dc_twist_endcaps(
                        twist=cfg_src["twist"], endcaps=float(cfg_src["endcaps"])
                    )
                    # Optional extra drive
                    if cfg_src.get("use_extra"):
                        try:
                            extra_map = json.loads(
                                cfg_src.get("extra_map_json") or "{}"
                            )
                        except Exception:
                            extra_map = {}
                        tv2.add_driving(
                            "ExtraDrive1", cfg_src["extra_freq"], 0.0, extra_map
                        )

                    with st.spinner("Solving F310 for requested c-slice targets…"):
                        result = solve_F310_for_targets_tensor(
                            num_ions=int(cfg_src["num_ions"]),
                            constant_trappingvars=tv2,
                            target_tensor_Hz=target,
                            bounds5=[(-1.0, 1.0)] * 5,
                            point=[0.0] * 5,  # linearization at origin (mirrors F110)
                            simulation_preset=cfg_src["preset"],
                            rest_penalty=0.0,
                            clip_to_bounds=False,
                        )

                    # Show solution 5-vector (symmetric DC channels)
                    st.success(
                        "Solved inputs (symmetric DC DOFs: DC1=10, DC2=9, DC3=8, DC4=7, DC5=6)"
                    )
                    u = result["u_solution_V"]
                    cols = st.columns(5)
                    for k, lab in enumerate(
                        ["(1=10)", "(2=9)", "(3=8)", "(4=7)", "(5=6)"]
                    ):
                        cols[k].metric(f"u[{k+1}] {lab}", f"{u[k]:.6f} V")

                    # Achieved vs target for just the edited c-slice
                    G_pred = result["G_pred_Hz"]
                    rows = []
                    for i, j, tgt in nonempty:
                        rows.append(
                            {
                                "triple (i,j,c)": f"({i},{j},{c_idx})",
                                "target (Hz)": tgt,
                                "achieved (Hz)": G_pred[i, j, c_idx],
                                "error (Hz)": G_pred[i, j, c_idx] - tgt,
                            }
                        )
                    st.write("Achieved vs target (this c-slice):")
                    st.dataframe(
                        pd.DataFrame(rows).style.format(
                            {
                                "target (Hz)": "{:.6e}",
                                "achieved (Hz)": "{:.6e}",
                                "error (Hz)": "{:.6e}",
                            }
                        ),
                        use_container_width=True,
                    )

                    st.caption(
                        f"Residual RMS over selected rows: {result['residual_rms']:.3e} Hz"
                    )

                except Exception as e:
                    st.error(f"F310 target solve failed: {e}")

    # ────────────────────────────────────────────────────────────────────────────
    # F120 Max Coupling Matrix (bounds ±.5 V on 20 channels)
    # ────────────────────────────────────────────────────────────────────────────
    with st.expander("F120 Max Coupling Matrix (bounds ±1 V on 20 channels)"):
        st.write(
            "Click **Run** to compute the per-pair max |g₀| (Hz) within ±0.5 V on each of the "
            "**20 independent channels** (DC1..DC10, RF11..RF20). Results persist until **Clear**."
        )
        cA, cB = st.columns([1, 1])
        run_f120_max = cA.button("Run F120 Max", key="btn_run_f120_max")
        clear_f120_max = cB.button("Clear", key="btn_clear_f120_max")

        if clear_f120_max:
            st.session_state.pop("f120_max_matrix", None)
            st.session_state.pop("f120_max_meta", None)

        if run_f120_max:
            cfg_src = cfg_cached if cfg_cached is not None else pending_cfg
            try:
                SimulationCls, Trapping_VarsCls = _ensure_imports(
                    cfg_src["repo_path"], cfg_src["add_repo_to_sys_path"]
                )
                tv2 = Trapping_VarsCls()
                # RF drive
                tv2.add_driving(
                    "RF",
                    cfg_src["rf_freq"],
                    0.0,
                    {"RF1": cfg_src["rf_amp1"], "RF2": cfg_src["rf_amp2"]},
                )
                # Manual DC offsets (before twist/endcaps)
                if cfg_src.get("manual_dc_enabled") and cfg_src.get(
                    "manual_dc_offsets"
                ):
                    ea2 = tv2.Var_dict[tv2.dc_key]
                    for el, V in cfg_src["manual_dc_offsets"].items():
                        try:
                            ea2.add_amplitude_volt(el, float(V))
                        except Exception:
                            pass
                # Geometry knobs
                tv2.apply_dc_twist_endcaps(
                    twist=cfg_src["twist"], endcaps=float(cfg_src["endcaps"])
                )
                # Optional extra drive
                if cfg_src.get("use_extra"):
                    try:
                        extra_map = json.loads(cfg_src.get("extra_map_json") or "{}")
                    except Exception:
                        extra_map = {}
                    tv2.add_driving(
                        "ExtraDrive1", cfg_src["extra_freq"], 0.0, extra_map
                    )

                with st.spinner("Computing F120 max matrix…"):
                    f120 = find_max_coupling_matrix_F120(
                        num_ions=int(cfg_src["num_ions"]),
                        constant_trappingvars=tv2,
                        point=[0.0] * 20,  # evaluate at origin in 20-D
                        amp_bounds=0.5,  # ±1 V on each channel
                        simulation_preset=cfg_src["preset"],
                    )
                st.session_state["f120_max_matrix"] = f120["Gmax_abs_Hz"]
                st.session_state["f120_max_meta"] = {
                    "N": int(cfg_src["num_ions"]),
                    "preset": cfg_src["preset"],
                    "bounds": "±1 V (20 channels)",
                }
                st.success("F120 max matrix computed.")
            except Exception as e:
                st.error(f"F120 max computation failed: {e}")

        if "f120_max_matrix" in st.session_state:
            Mmax = st.session_state["f120_max_matrix"]
            meta = st.session_state.get("f120_max_meta", {})

            st.caption(
                f"Upper-triangular matrix of per-pair maxima (Hz). "
                f"N={meta.get('N','?')}, preset={meta.get('preset','?')}, bounds={meta.get('bounds','?')}."
            )

            df = pd.DataFrame(Mmax)
            vmax = float(np.nanmax(Mmax)) if np.size(Mmax) else 0.0
            vmin_visible = 1.0  # show ≥1 Hz as visible blue

            def _cell_color_log(val, _vmin=vmin_visible, _vmax=vmax):
                try:
                    x = float(val)
                except Exception:
                    return "background-color: #0e0f13; color: #eaeaea"
                if not np.isfinite(x) or x <= 0.0:
                    return "background-color: #0e0f13; color: #eaeaea"
                if _vmax <= _vmin:
                    t = 1.0
                else:
                    t = (np.log10(x) - np.log10(_vmin)) / (
                        np.log10(_vmax) - np.log10(_vmin)
                    )
                    t = min(1.0, max(0.0, t))
                b = int(round(255 * t))
                return f"background-color: rgb(0,0,{b}); color: #eaeaea"

            styled = df.style.format("{:.3e}").applymap(_cell_color_log)
            st.dataframe(styled, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────────
    # F120 Targeted Coupling Solver (enter desired g₀ targets, Hz)
    # ────────────────────────────────────────────────────────────────────────────
    with st.expander("F120 Targeted Coupling Solver (enter desired g₀ targets, Hz)"):
        st.write(
            "Enter a **K×K** matrix of desired g₀ couplings (Hz). "
            "Only the **upper triangle** (i<j) is used; diagonal and lower triangle are ignored. "
            "Then click **Compute** to solve for the **20 independent inputs** (DC1..DC10, RF11..RF20)."
        )

        K = 3 * int(num_ions)
        key_df = f"f120_target_matrix_K{K}"
        if key_df not in st.session_state:
            st.session_state[key_df] = pd.DataFrame(
                np.zeros((K, K), dtype=float),
                index=[f"i={i}" for i in range(K)],
                columns=[f"j={j}" for j in range(K)],
            )

        df_target = st.data_editor(
            st.session_state[key_df],
            use_container_width=True,
            num_rows="dynamic",
            key=f"editor_{key_df}",
        )

        c1, c2, c3 = st.columns([1, 1, 2])
        compute_btn = c1.button("Compute inputs (F120)", key="btn_f120_targets_compute")
        clear_btn = c2.button("Clear matrix", key="btn_f120_targets_clear")
        c3.caption("Tip: fill only cells with i<j. Others will be ignored.")

        if clear_btn:
            st.session_state[key_df] = pd.DataFrame(
                np.zeros((K, K), dtype=float),
                index=[f"i={i}" for i in range(K)],
                columns=[f"j={j}" for j in range(K)],
            )
            df_target = st.session_state[key_df]

        if compute_btn:
            targets = []
            for i in range(K):
                for j in range(i + 1, K):
                    try:
                        val = float(df_target.iat[i, j])
                    except Exception:
                        val = 0.0
                    if val != 0.0:
                        targets.append(((i, j), val))

            if not targets:
                st.warning(
                    "No upper-triangle targets entered. Fill some (i<j) entries and try again."
                )
            else:
                cfg_src = cfg_cached if cfg_cached is not None else pending_cfg
                try:
                    SimulationCls, Trapping_VarsCls = _ensure_imports(
                        cfg_src["repo_path"], cfg_src["add_repo_to_sys_path"]
                    )
                    tv2 = Trapping_VarsCls()
                    # RF
                    tv2.add_driving(
                        "RF",
                        cfg_src["rf_freq"],
                        0.0,
                        {"RF1": cfg_src["rf_amp1"], "RF2": cfg_src["rf_amp2"]},
                    )
                    # Manual DC → DC drive before twist/endcaps
                    if cfg_src.get("manual_dc_enabled") and cfg_src.get(
                        "manual_dc_offsets"
                    ):
                        ea2 = tv2.Var_dict[tv2.dc_key]
                        for el, V in cfg_src["manual_dc_offsets"].items():
                            try:
                                ea2.add_amplitude_volt(el, float(V))
                            except Exception:
                                pass
                    # Geometry
                    tv2.apply_dc_twist_endcaps(
                        twist=cfg_src["twist"], endcaps=float(cfg_src["endcaps"])
                    )
                    # Optional extra drive
                    if cfg_src.get("use_extra"):
                        try:
                            extra_map = json.loads(
                                cfg_src.get("extra_map_json") or "{}"
                            )
                        except Exception:
                            extra_map = {}
                        tv2.add_driving(
                            "ExtraDrive1", cfg_src["extra_freq"], 0.0, extra_map
                        )

                    with st.spinner("Solving F120 for requested targets…"):
                        result = solve_F120_for_targets(
                            mode_pair_targets=targets,  # [((i,j), Hz), ...]
                            num_ions=int(cfg_src["num_ions"]),
                            constant_trappingvars=tv2,
                            point=[0.0] * 20,  # linearization at origin
                            amp_bounds=1.0,  # ±1 V on each channel
                            l2_reg=0.0,
                            minimize_rest=False,
                            clip_to_bounds=False,
                            simulation_preset=cfg_src["preset"],
                        )

                    # Show 20-channel solution neatly
                    st.success("Solved inputs (20 channels: DC1..DC10, RF11..RF20)")
                    order = [
                        "DC1",
                        "DC2",
                        "DC3",
                        "DC4",
                        "DC5",
                        "DC6",
                        "DC7",
                        "DC8",
                        "DC9",
                        "DC10",
                        "RF11",
                        "RF12",
                        "RF13",
                        "RF14",
                        "RF15",
                        "RF16",
                        "RF17",
                        "RF18",
                        "RF19",
                        "RF20",
                    ]
                    x = result["amplitudes_vector"]
                    rows = [
                        {"channel": name, "V": float(x[k])}
                        for k, name in enumerate(order)
                    ]
                    st.dataframe(
                        pd.DataFrame(rows).style.format({"V": "{:.6f}"}),
                        use_container_width=True,
                        height=360,
                    )

                    # Achieved vs target table
                    pred = dict(result["predicted_targets_Hz"])
                    rows = []
                    for (i, j), tgt in targets:
                        rows.append(
                            {
                                "pair (i,j)": f"({i},{j})",
                                "target (Hz)": tgt,
                                "achieved (Hz)": pred[(i, j)],
                                "error (Hz)": pred[(i, j)] - tgt,
                            }
                        )
                    st.write("Achieved vs target:")
                    st.dataframe(
                        pd.DataFrame(rows).style.format(
                            {
                                "target (Hz)": "{:.6e}",
                                "achieved (Hz)": "{:.6e}",
                                "error (Hz)": "{:.6e}",
                            }
                        ),
                        use_container_width=True,
                    )

                    st.caption(
                        f"Predicted rest RMS (over non-target rows): {result['predicted_rest_rms_Hz']:.3e} Hz"
                    )

                except Exception as e:
                    st.error(f"F120 target solve failed: {e}")

    # ────────────────────────────────────────────────────────────────────────────
    # F320 Max Coupling Tensor (bounds ±1 V on 20 channels)
    # ────────────────────────────────────────────────────────────────────────────
    with st.expander("F320 Max Coupling Tensor (bounds ±1 V on 20 channels)"):
        st.write(
            "Click **Run** to compute the per-(a<b, c) max |g₀| (Hz) within ±1 V "
            "on each of the **20 independent channels** (DC1..DC10, RF11..RF20). "
            "Results persist until **Clear**."
        )
        cA, cB = st.columns([1, 1])
        run_f320_max = cA.button("Run F320 Max", key="btn_run_f320_max")
        clear_f320_max = cB.button("Clear", key="btn_clear_f320_max")

        if clear_f320_max:
            st.session_state.pop("f320_max_tensor", None)
            st.session_state.pop("f320_max_meta", None)

        if run_f320_max:
            cfg_src = cfg_cached if cfg_cached is not None else pending_cfg
            try:
                SimulationCls, Trapping_VarsCls = _ensure_imports(
                    cfg_src["repo_path"], cfg_src["add_repo_to_sys_path"]
                )
                tv2 = Trapping_VarsCls()
                # RF drive
                tv2.add_driving(
                    "RF",
                    cfg_src["rf_freq"],
                    0.0,
                    {"RF1": cfg_src["rf_amp1"], "RF2": cfg_src["rf_amp2"]},
                )
                # Manual DC offsets (before twist/endcaps)
                if cfg_src.get("manual_dc_enabled") and cfg_src.get(
                    "manual_dc_offsets"
                ):
                    ea2 = tv2.Var_dict[tv2.dc_key]
                    for el, V in cfg_src["manual_dc_offsets"].items():
                        try:
                            ea2.add_amplitude_volt(el, float(V))
                        except Exception:
                            pass
                # Geometry knobs
                tv2.apply_dc_twist_endcaps(
                    twist=cfg_src["twist"], endcaps=float(cfg_src["endcaps"])
                )
                # Optional extra drive (parity with main sim)
                if cfg_src.get("use_extra"):
                    try:
                        extra_map = json.loads(cfg_src.get("extra_map_json") or "{}")
                    except Exception:
                        extra_map = {}
                    tv2.add_driving(
                        "ExtraDrive1", cfg_src["extra_freq"], 0.0, extra_map
                    )

                with st.spinner("Computing F320 max tensor…"):
                    f320 = find_max_coupling_matrix_F320(
                        num_ions=int(cfg_src["num_ions"]),
                        constant_trappingvars=tv2,
                        point=[0.0] * 20,  # evaluate at origin in 20-D
                        amp_bounds=1.0,  # ±1 V on each channel
                        simulation_preset=cfg_src["preset"],
                    )
                # Enforce full permutation symmetry for display (fill all permutations)
                Gmax_sym = _symmetrize_rank3_abs(f320["G3max_abs_Hz"])
                st.session_state["f320_max_tensor"] = Gmax_sym
                st.session_state["f320_max_meta"] = {
                    "N": int(cfg_src["num_ions"]),
                    "preset": cfg_src["preset"],
                    "bounds": "±1 V (20 channels)",
                    "K": int(Gmax_sym.shape[0]),
                }
                st.success("F320 max tensor computed.")
            except Exception as e:
                st.error(f"F320 max computation failed: {e}")

        # Display (slice by c)
        if "f320_max_tensor" in st.session_state:
            Gmax = st.session_state["f320_max_tensor"]
            meta = st.session_state.get("f320_max_meta", {})
            K = int(meta.get("K", Gmax.shape[0]))
            st.caption(
                f"Tensor slices (Hz). Showing Gmax[:, :, c]. "
                f"N={meta.get('N','?')}, preset={meta.get('preset','?')}, bounds={meta.get('bounds','?')}."
            )
            c_idx = st.selectbox(
                "Select c-index (mode)",
                options=list(range(K)),
                index=0,
                key="f320_max_cidx",
            )
            slice_mat = np.asarray(Gmax)[:, :, c_idx]
            df_slice = pd.DataFrame(slice_mat)

            def _style_upper_log(df: pd.DataFrame) -> pd.DataFrame:
                """
                Color ONLY the upper triangle **including the diagonal** (i<=j).
                Log-scale is computed from strictly-positive values on that region.
                Zeros / non-positives / NaNs are treated as 'no signal' (dark).
                """
                arr = df.to_numpy(copy=False)
                n = arr.shape[0]
                css = np.full(arr.shape, "", dtype=object)
                iu = np.triu_indices(n, k=0)
                upper_vals = arr[iu]
                pos = upper_vals[(upper_vals > 0) & np.isfinite(upper_vals)]
                if pos.size == 0:
                    return pd.DataFrame(css, index=df.index, columns=df.columns)
                vmin = float(pos.min())
                vmax = float(pos.max())
                log_min = math.log10(vmin)
                log_max = math.log10(vmax) if vmax > vmin else log_min + 1e-9
                for i, j in zip(*iu):
                    x = arr[i, j]
                    if not np.isfinite(x) or x <= 0.0:
                        css[i, j] = "background-color: #0e0f13; color: #eaeaea"
                    else:
                        t = (math.log10(x) - log_min) / (log_max - log_min)
                        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                        b = int(round(255 * t))
                        css[i, j] = f"background-color: rgb(0,0,{b}); color: #eaeaea"
                return pd.DataFrame(css, index=df.index, columns=df.columns)

            styled = df_slice.style.format("{:.3e}").apply(_style_upper_log, axis=None)
            st.dataframe(styled, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────────────
    # F320 Targeted Coupling Solver (3-mode) — like F310 but with 20 independent inputs
    # ────────────────────────────────────────────────────────────────────────────
    with st.expander("F320 Targeted Coupling Solver (enter desired g₀ targets, Hz)"):
        st.write(
            "Pick a **c-index** (mode) and edit the **K×K** table for that slice. "
            "Only the **upper triangle** (i<j) is used. Click **Compute** to solve for the "
            "**20 independent inputs** (DC1..DC10, RF11..RF20)."
        )

        K = 3 * int(num_ions)
        c_idx = st.selectbox(
            "Select c-index (mode for the slice)",
            options=list(range(K)),
            index=0,
            key="f320_target_cidx",
        )

        key_df = f"f320_target_slice_c{c_idx}_K{K}"
        if key_df not in st.session_state:
            st.session_state[key_df] = pd.DataFrame(
                np.zeros((K, K), dtype=float),
                index=[f"i={i}" for i in range(K)],
                columns=[f"j={j}" for j in range(K)],
            )

        df_target_slice = st.data_editor(
            st.session_state[key_df],
            use_container_width=True,
            num_rows="dynamic",
            key=f"editor_{key_df}",
        )

        c1, c2, c3 = st.columns([1, 1, 2])
        compute_btn = c1.button("Compute inputs (F320)", key="btn_f320_targets_compute")
        clear_btn = c2.button("Clear slice", key="btn_f320_targets_clear")
        c3.caption(
            "Tip: fill only cells with i<j. Others are ignored. Leave 0.0 to skip a pair."
        )

        if clear_btn:
            st.session_state[key_df] = pd.DataFrame(
                np.zeros((K, K), dtype=float),
                index=[f"i={i}" for i in range(K)],
                columns=[f"j={j}" for j in range(K)],
            )
            df_target_slice = st.session_state[key_df]

        if compute_btn:
            # Build target triples for this c-slice (only nonzero upper-triangle entries)
            targets = []
            nonempty = []
            for i in range(K):
                for j in range(i + 1, K):
                    try:
                        val = float(df_target_slice.iat[i, j])
                    except Exception:
                        val = 0.0
                    if val != 0.0:
                        targets.append(((i, j, c_idx), val))
                        nonempty.append((i, j, val))

            if not targets:
                st.warning(
                    "No upper-triangle targets entered for this c-slice. Fill some (i<j) and try again."
                )
            else:
                cfg_src = cfg_cached if cfg_cached is not None else pending_cfg
                try:
                    SimulationCls, Trapping_VarsCls = _ensure_imports(
                        cfg_src["repo_path"], cfg_src["add_repo_to_sys_path"]
                    )
                    tv2 = Trapping_VarsCls()
                    # RF
                    tv2.add_driving(
                        "RF",
                        cfg_src["rf_freq"],
                        0.0,
                        {"RF1": cfg_src["rf_amp1"], "RF2": cfg_src["rf_amp2"]},
                    )
                    # Manual DC → DC drive before twist/endcaps
                    if cfg_src.get("manual_dc_enabled") and cfg_src.get(
                        "manual_dc_offsets"
                    ):
                        ea2 = tv2.Var_dict[tv2.dc_key]
                        for el, V in cfg_src["manual_dc_offsets"].items():
                            try:
                                ea2.add_amplitude_volt(el, float(V))
                            except Exception:
                                pass
                    # Geometry knobs
                    tv2.apply_dc_twist_endcaps(
                        twist=cfg_src["twist"], endcaps=float(cfg_src["endcaps"])
                    )
                    # Optional extra drive
                    if cfg_src.get("use_extra"):
                        try:
                            extra_map = json.loads(
                                cfg_src.get("extra_map_json") or "{}"
                            )
                        except Exception:
                            extra_map = {}
                        tv2.add_driving(
                            "ExtraDrive1", cfg_src["extra_freq"], 0.0, extra_map
                        )

                    with st.spinner("Solving F320 for requested c-slice targets…"):
                        result = solve_F320_for_targets(
                            mode_triple_targets=targets,  # [((i,j,c), Hz), ...]
                            num_ions=int(cfg_src["num_ions"]),
                            constant_trappingvars=tv2,
                            point=[0.0] * 20,  # linearization at origin
                            amp_bounds=1.0,  # ±1 V on each channel
                            l2_reg=0.0,
                            minimize_rest=False,
                            clip_to_bounds=False,
                            simulation_preset=cfg_src["preset"],
                        )

                    # Show 20-channel solution neatly
                    st.success("Solved inputs (20 channels: DC1..DC10, RF11..RF20)")
                    order = [
                        "DC1",
                        "DC2",
                        "DC3",
                        "DC4",
                        "DC5",
                        "DC6",
                        "DC7",
                        "DC8",
                        "DC9",
                        "DC10",
                        "RF11",
                        "RF12",
                        "RF13",
                        "RF14",
                        "RF15",
                        "RF16",
                        "RF17",
                        "RF18",
                        "RF19",
                        "RF20",
                    ]
                    x = result["amplitudes_vector"]
                    rows = [
                        {"channel": name, "V": float(x[k])}
                        for k, name in enumerate(order)
                    ]
                    st.dataframe(
                        pd.DataFrame(rows).style.format({"V": "{:.6f}"}),
                        use_container_width=True,
                        height=360,
                    )

                    # Achieved vs target for just the edited c-slice
                    G_pred = result["predicted_tensor_Hz"]  # K×K×K with a<b filled
                    rows = []
                    for i, j, tgt in nonempty:
                        rows.append(
                            {
                                "triple (i,j,c)": f"({i},{j},{c_idx})",
                                "target (Hz)": tgt,
                                "achieved (Hz)": G_pred[i, j, c_idx],
                                "error (Hz)": G_pred[i, j, c_idx] - tgt,
                            }
                        )
                    st.write("Achieved vs target (this c-slice):")
                    st.dataframe(
                        pd.DataFrame(rows).style.format(
                            {
                                "target (Hz)": "{:.6e}",
                                "achieved (Hz)": "{:.6e}",
                                "error (Hz)": "{:.6e}",
                            }
                        ),
                        use_container_width=True,
                    )

                    st.caption(
                        f"Residual RMS over non-target rows: {result['predicted_rest_rms_Hz']:.3e} Hz"
                    )

                except Exception as e:
                    st.error(f"F320 target solve failed: {e}")

    st.divider()
    st.subheader("Raw resonance JSON (for debugging / export)")
    st.code(json.dumps(_to_jsonable(res["resonances"]), indent=2))

    st.download_button(
        label="Download frequencies.json",
        data=json.dumps(_to_jsonable(res["frequencies_Hz"]), indent=2),
        file_name="frequencies.json",
        mime="application/json",
    )
    st.download_button(
        label="Download resonances.json",
        data=json.dumps(_to_jsonable(res["resonances"]), indent=2),
        file_name="resonances.json",
        mime="application/json",
    )

else:
    st.info("Set parameters in the sidebar and click **Compute**.")
