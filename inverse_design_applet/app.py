from __future__ import annotations

import ast
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import constants
from control_constraints import rotation_from_axis_ref_alpha
from inverse_design import solve_u_for_frequency_box
from reachability import (
    DEFAULT_DEDUPLICATE_TOL,
    build_reachability_model,
    plot_lambda_boundary_points_3d_plotly,
    plot_multi_trap_frequency_space,
    plot_single_trap_frequency_space,
    sample_reachable_boundary,
)
from sim.simulation import Simulation
from trapping_variables import Trapping_Vars

APP_MODEL_NUM_SAMPLES = 80
APP_POLYFIT_DEG = 4
APP_RANDOM_SEED = 1
APP_DENSITY_SCALE = 1.4
REACHABILITY_SAMPLE_DEFAULT = 2000
MULTI_SPEC_SAMPLE_DEFAULT = 2000
DEFAULT_RF_FREQ_MHZ = constants.RF_FREQ_REF_HZ / 1.0e6
DEFAULT_RF_AMP_V = math.sqrt(constants.RF_S_MAX_DEFAULT) * constants.RF_OMEGA_REF_MHZ

REACH_SINGLE_RESULT_KEY = "inverse_app_reach_single_result"
REACH_SINGLE_ERROR_KEY = "inverse_app_reach_single_error"
REACH_MULTI_RESULT_KEY = "inverse_app_reach_multi_result"
REACH_MULTI_ERROR_KEY = "inverse_app_reach_multi_error"
INVERSE_RESULT_KEY = "inverse_app_inverse_result"
INVERSE_ERROR_KEY = "inverse_app_inverse_error"

PRINCIPAL_AXIS_HELP = (
    "Direction for the first principal mode. Enter a 3-vector such as [1, 0, 0]. "
    "The solver normalizes it."
)
REF_DIR_HELP = (
    "Reference direction used to orient the other two principal modes. "
    "It must not be parallel to principal_axis."
)
ALPHA_HELP = (
    "Rotation angle in degrees for the second principal direction after ref_dir is "
    "projected into the plane perpendicular to principal_axis."
)

MULTI_SPEC_EXAMPLE = """[
    {
        "display_label": "2Dtrap_125_45deg_200exp",
        "name": "2D +/-100",
        "alpha_deg": 0.0,
        "u_bounds": [(-100.0, 100.0)] * 22 + [(0.0, RF_S_MAX_DEFAULT)],
    },
    {
        "display_label": "InnTrapFine",
        "name": "InnTrapFine",
        "alpha_deg": 0.0,
        "u_bounds": [(-100.0, 100.0)] * 14 + [(0.0, RF_S_MAX_DEFAULT)],
    },
]"""

CONTROL_GRID_STYLE = """
<style>
.inverse-app-grid-title {
    margin: 0.5rem 0 0.4rem 0;
    font-weight: 600;
}
.inverse-app-grid {
    display: grid;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.inverse-app-grid.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.inverse-app-grid.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.inverse-app-grid.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.inverse-app-grid.cols-5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.inverse-app-grid.cols-6 { grid-template-columns: repeat(6, minmax(0, 1fr)); }
.inverse-app-grid.cols-7 { grid-template-columns: repeat(7, minmax(0, 1fr)); }
.inverse-app-cell {
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem;
    padding: 0.5rem 0.6rem;
    background: rgba(248, 249, 251, 0.7);
}
.inverse-app-cell.empty {
    border: none;
    background: transparent;
}
.inverse-app-name {
    font-size: 0.8rem;
    color: rgb(49, 51, 63);
}
.inverse-app-value {
    font-family: "Courier New", monospace;
    font-size: 0.95rem;
    margin-top: 0.15rem;
}
</style>
"""


def main() -> None:
    st.set_page_config(page_title="Inverse Design Applet", layout="wide")
    ensure_session_state()

    st.title("Inverse Design Applet")
    st.caption(
        "Thin Streamlit wrapper for the existing inverse-design and reachability pipeline. "
        "Trap selection changes which cached A-model is used; cache controls stay hidden."
    )

    with st.expander("Reachability Workspace", expanded=True):
        render_reachability_workspace()

    with st.expander("Inverse Solve Workspace", expanded=True):
        render_inverse_workspace()


def ensure_session_state() -> None:
    for key in (
        REACH_SINGLE_RESULT_KEY,
        REACH_MULTI_RESULT_KEY,
        INVERSE_RESULT_KEY,
        REACH_SINGLE_ERROR_KEY,
        REACH_MULTI_ERROR_KEY,
        INVERSE_ERROR_KEY,
    ):
        if key not in st.session_state:
            st.session_state[key] = None


def render_reachability_workspace() -> None:
    trap_labels = list(constants.INVERSE_APP_TRAP_OPTIONS)

    with st.form("reachability_single_form"):
        st.subheader("Single-Trap Reachability")
        trap_label = st.selectbox("Trap", options=trap_labels, key="reach_trap_label")

        bounds_col, rf_freq_col, rf_amp_col = st.columns(3)
        dc_bound_text = bounds_col.text_input(
            "Abs bound on DC controls",
            value="",
            placeholder="100",
        )
        ref_rf_freq_mhz = rf_freq_col.number_input(
            "Reference RF frequency (MHz)",
            min_value=0.001,
            value=float(DEFAULT_RF_FREQ_MHZ),
            step=0.1,
            format="%.6f",
        )
        max_rf_amp_v = rf_amp_col.number_input(
            "Max RF amplitude at reference frequency (V)",
            min_value=0.0,
            value=float(DEFAULT_RF_AMP_V),
            step=10.0,
            format="%.6f",
        )

        x_col, y_col, z_col = st.columns(3)
        x_m = x_col.number_input("x (m)", value=0.0, format="%.9g")
        y_m = y_col.number_input("y (m)", value=0.0, format="%.9g")
        z_m = z_col.number_input("z (m)", value=0.0, format="%.9g")

        axis_col, ref_col, alpha_col = st.columns(3)
        principal_axis_text = axis_col.text_input(
            "principal_axis",
            value="",
            placeholder="[1, 0, 0]",
            help=PRINCIPAL_AXIS_HELP,
        )
        ref_dir_text = ref_col.text_input(
            "ref_dir",
            value="",
            placeholder="[0, 0, 1]",
            help=REF_DIR_HELP,
        )
        alpha_deg_text = alpha_col.text_input(
            "alpha_deg",
            value="",
            placeholder="0",
            help=ALPHA_HELP,
        )

        sample_col, lambda_col = st.columns([1, 1])
        n_samples = sample_col.number_input(
            "Reachability sample points",
            min_value=100,
            step=100,
            value=REACHABILITY_SAMPLE_DEFAULT,
        )
        show_lambda = lambda_col.checkbox("Also show lambda-space plot", value=False)

        run_single = st.form_submit_button("Run Single-Trap Reachability", type="primary")

    if run_single:
        try:
            result = run_single_reachability(
                trap_label=trap_label,
                dc_bound_abs=parse_required_float(dc_bound_text, "Abs bound on DC controls", min_value=0.0),
                ref_rf_freq_hz=float(ref_rf_freq_mhz) * 1.0e6,
                max_rf_amp_v=float(max_rf_amp_v),
                r0=np.array([x_m, y_m, z_m], dtype=float),
                principal_axis_text=principal_axis_text,
                ref_dir_text=ref_dir_text,
                alpha_deg_text=alpha_deg_text,
                n_samples=int(n_samples),
                show_lambda=bool(show_lambda),
            )
        except Exception as exc:
            st.session_state[REACH_SINGLE_ERROR_KEY] = str(exc)
        else:
            st.session_state[REACH_SINGLE_RESULT_KEY] = result
            st.session_state[REACH_SINGLE_ERROR_KEY] = None

    render_workspace_message(REACH_SINGLE_ERROR_KEY)
    render_single_reachability_result(st.session_state[REACH_SINGLE_RESULT_KEY])

    with st.expander("Multi-Spec Reachability Overlay", expanded=False):
        with st.form("reachability_multi_form"):
            st.caption(
                "Paste a Python-literal-style list/dict spec. A narrow Python-style fallback "
                "is also accepted for demo-like expressions such as `* 22` and `RF_S_MAX_DEFAULT`."
            )
            multi_sample_count = st.number_input(
                "Reachability sample points per spec",
                min_value=100,
                step=100,
                value=MULTI_SPEC_SAMPLE_DEFAULT,
            )
            raw_specs_text = st.text_area(
                "Raw multi-spec textbox",
                value=MULTI_SPEC_EXAMPLE,
                height=260,
            )
            run_multi = st.form_submit_button("Run Multi-Spec Overlay")

        if run_multi:
            try:
                result = run_multi_spec_overlay(
                    raw_specs_text=raw_specs_text,
                    n_samples=int(multi_sample_count),
                )
            except Exception as exc:
                st.session_state[REACH_MULTI_ERROR_KEY] = str(exc)
            else:
                st.session_state[REACH_MULTI_RESULT_KEY] = result
                st.session_state[REACH_MULTI_ERROR_KEY] = None

        render_workspace_message(REACH_MULTI_ERROR_KEY)
        render_multi_reachability_result(st.session_state[REACH_MULTI_RESULT_KEY])


def render_inverse_workspace() -> None:
    trap_labels = list(constants.INVERSE_APP_TRAP_OPTIONS)
    objective_labels = (
        "L2 min for all",
        "L2 min for all but s",
        "L2_Sx100",
        "s_min",
        "s_max",
    )

    with st.form("inverse_solve_form"):
        trap_label = st.selectbox("Trap", options=trap_labels, key="inverse_trap_label")

        x_col, y_col, z_col = st.columns(3)
        x_m = x_col.number_input("x (m)", value=0.0, format="%.9g", key="inverse_x")
        y_m = y_col.number_input("y (m)", value=0.0, format="%.9g", key="inverse_y")
        z_m = z_col.number_input("z (m)", value=0.0, format="%.9g", key="inverse_z")

        axis_col, ref_col, alpha_col = st.columns(3)
        principal_axis_text = axis_col.text_input(
            "principal_axis",
            value="",
            placeholder="[1, 0, 0]",
            help=PRINCIPAL_AXIS_HELP,
            key="inverse_principal_axis",
        )
        ref_dir_text = ref_col.text_input(
            "ref_dir",
            value="",
            placeholder="[0, 0, 1]",
            help=REF_DIR_HELP,
            key="inverse_ref_dir",
        )
        alpha_deg_text = alpha_col.text_input(
            "alpha_deg",
            value="",
            placeholder="0",
            help=ALPHA_HELP,
            key="inverse_alpha_deg",
        )

        st.markdown("**Desired frequency bounds (MHz)**")
        header_cols = st.columns(3)
        header_cols[0].markdown("Mode 1")
        header_cols[1].markdown("Mode 2")
        header_cols[2].markdown("Mode 3")
        lower_cols = st.columns(3)
        upper_cols = st.columns(3)
        f1_lo = lower_cols[0].text_input("Lower", value="", placeholder="0.300", key="inverse_f1_lo")
        f2_lo = lower_cols[1].text_input("Lower", value="", placeholder="0.600", key="inverse_f2_lo")
        f3_lo = lower_cols[2].text_input("Lower", value="", placeholder="0.900", key="inverse_f3_lo")
        f1_hi = upper_cols[0].text_input("Upper", value="", placeholder="0.300", key="inverse_f1_hi")
        f2_hi = upper_cols[1].text_input("Upper", value="", placeholder="0.600", key="inverse_f2_hi")
        f3_hi = upper_cols[2].text_input("Upper", value="", placeholder="0.900", key="inverse_f3_hi")

        objective_col, dc_col, rf_dc_col = st.columns(3)
        objective_label = objective_col.selectbox("Minimization routine", options=objective_labels)
        dc_bound_text = dc_col.text_input(
            "Abs bound on DC controls",
            value="",
            placeholder="100",
            key="inverse_dc_bound",
        )
        rf_dc_bound_text = rf_dc_col.text_input(
            "Abs bound on RF-DC controls",
            value="",
            placeholder="100",
            key="inverse_rf_dc_bound",
        )

        rf_freq_col, rf_amp_col = st.columns(2)
        ref_rf_freq_mhz = rf_freq_col.number_input(
            "Reference RF frequency (MHz)",
            min_value=0.001,
            value=float(DEFAULT_RF_FREQ_MHZ),
            step=0.1,
            format="%.6f",
            key="inverse_ref_freq",
        )
        max_rf_amp_v = rf_amp_col.number_input(
            "Max RF amplitude at reference frequency (V)",
            min_value=0.0,
            value=float(DEFAULT_RF_AMP_V),
            step=10.0,
            format="%.6f",
            key="inverse_rf_amp",
        )

        run_inverse = st.form_submit_button("Run Bounded Inverse Solve", type="primary")

    if run_inverse:
        try:
            result = run_inverse_solve(
                trap_label=trap_label,
                r0=np.array([x_m, y_m, z_m], dtype=float),
                principal_axis_text=principal_axis_text,
                ref_dir_text=ref_dir_text,
                alpha_deg_text=alpha_deg_text,
                freq_bound_texts=((f1_lo, f1_hi), (f2_lo, f2_hi), (f3_lo, f3_hi)),
                objective_label=objective_label,
                dc_bound_abs=parse_required_float(dc_bound_text, "Abs bound on DC controls", min_value=0.0),
                rf_dc_bound_abs=parse_required_float(
                    rf_dc_bound_text,
                    "Abs bound on RF-DC controls",
                    min_value=0.0,
                ),
                ref_rf_freq_hz=float(ref_rf_freq_mhz) * 1.0e6,
                max_rf_amp_v=float(max_rf_amp_v),
            )
        except Exception as exc:
            st.session_state[INVERSE_ERROR_KEY] = str(exc)
        else:
            st.session_state[INVERSE_RESULT_KEY] = result
            st.session_state[INVERSE_ERROR_KEY] = None

    render_workspace_message(INVERSE_ERROR_KEY)
    render_inverse_result(st.session_state[INVERSE_RESULT_KEY])


def render_workspace_message(key: str) -> None:
    message = st.session_state.get(key)
    if message:
        st.error(message)


def run_single_reachability(
    *,
    trap_label: str,
    dc_bound_abs: float,
    ref_rf_freq_hz: float,
    max_rf_amp_v: float,
    r0: np.ndarray,
    principal_axis_text: str,
    ref_dir_text: str,
    alpha_deg_text: str,
    n_samples: int,
    show_lambda: bool,
) -> dict[str, Any]:
    trap_cfg = get_trap_registry_entry(trap_label)
    principal_axis, ref_dir, alpha_deg = parse_principal_direction_inputs(
        principal_axis_text,
        ref_dir_text,
        alpha_deg_text,
    )
    s_max = compute_s_max(ref_rf_freq_hz=ref_rf_freq_hz, max_rf_amp_v=max_rf_amp_v)
    u_bounds = build_u_bounds(
        n_dc=len(trap_cfg["dc_electrodes"]),
        n_rf=len(trap_cfg["rf_dc_electrodes"]),
        dc_bound_abs=dc_bound_abs,
        rf_dc_bound_abs=dc_bound_abs,
        s_max=s_max,
    )

    with st.spinner("Running reachability sampling..."):
        model = build_reachability_model(
            r0=r0,
            principal_axis=principal_axis,
            ref_dir=ref_dir,
            alpha_deg=alpha_deg,
            trap_name=str(trap_cfg["trap_name"]),
            dc_electrodes=list(trap_cfg["dc_electrodes"]),
            rf_dc_electrodes=list(trap_cfg["rf_dc_electrodes"]),
            num_samples=APP_MODEL_NUM_SAMPLES,
            u_bounds=u_bounds,
            polyfit_deg=APP_POLYFIT_DEG,
            seed=0,
            use_cache=True,
            ion_mass_kg=constants.ion_mass,
            ion_charge_c=constants.ion_charge,
            poly_is_potential_energy=False,
        )
        boundary = sample_reachable_boundary(
            model,
            n_samples=n_samples,
            random_seed=APP_RANDOM_SEED,
            deduplicate_tol=DEFAULT_DEDUPLICATE_TOL,
            build_hull=True,
        )
        frequency_fig, _, freq_sample = plot_single_trap_frequency_space(
            model,
            boundary,
            output="hz",
            density_scale=APP_DENSITY_SCALE,
            show_surface=True,
            backend="plotly",
            random_seed=APP_RANDOM_SEED,
            show=False,
            label=str(trap_cfg["display_label"]),
        )
        lambda_fig = None
        if show_lambda:
            lambda_fig = plot_lambda_boundary_points_3d_plotly(
                freq_sample.lambda_boundary.lambda_points,
                surface_points=freq_sample.lambda_boundary.surface_lambda_points,
                surface_triangles=freq_sample.lambda_boundary.surface_triangles,
                show_surface=True,
                show=False,
                label=str(trap_cfg["display_label"]),
            )

    return {
        "trap_label": trap_label,
        "trap_name": trap_cfg["trap_name"],
        "frequency_figure": frequency_fig,
        "lambda_figure": lambda_fig,
        "show_lambda": show_lambda,
        "cache_hit": bool(model.metadata.get("cache_hit", False)),
        "cache_path": model.metadata.get("cache_path"),
        "n_requested": int(boundary.n_requested),
        "n_success": int(boundary.n_success),
        "n_boundary_points": int(freq_sample.n_points),
        "s_max": s_max,
        "ref_rf_freq_hz": ref_rf_freq_hz,
        "max_rf_amp_v": max_rf_amp_v,
    }


def run_multi_spec_overlay(*, raw_specs_text: str, n_samples: int) -> dict[str, Any]:
    specs = parse_multi_spec_text(raw_specs_text)
    with st.spinner("Running multi-spec overlay..."):
        figure, _, results = plot_multi_trap_frequency_space(
            specs,
            n_samples=n_samples,
            num_model_samples=APP_MODEL_NUM_SAMPLES,
            random_seed=APP_RANDOM_SEED,
            density_scale=APP_DENSITY_SCALE,
            show_surface=True,
            plot_lambda_space=False,
            backend="plotly",
            output="hz",
            show=False,
        )
    return {
        "figure": figure,
        "n_specs": len(specs),
        "n_results": len(results),
        "n_samples": n_samples,
    }


def run_inverse_solve(
    *,
    trap_label: str,
    r0: np.ndarray,
    principal_axis_text: str,
    ref_dir_text: str,
    alpha_deg_text: str,
    freq_bound_texts: Sequence[tuple[str, str]],
    objective_label: str,
    dc_bound_abs: float,
    rf_dc_bound_abs: float,
    ref_rf_freq_hz: float,
    max_rf_amp_v: float,
) -> dict[str, Any]:
    trap_cfg = get_trap_registry_entry(trap_label)
    principal_axis, ref_dir, alpha_deg = parse_principal_direction_inputs(
        principal_axis_text,
        ref_dir_text,
        alpha_deg_text,
    )
    freq_bounds_hz = parse_frequency_bounds_mhz(freq_bound_texts)
    s_max = compute_s_max(ref_rf_freq_hz=ref_rf_freq_hz, max_rf_amp_v=max_rf_amp_v)
    u_bounds = build_u_bounds(
        n_dc=len(trap_cfg["dc_electrodes"]),
        n_rf=len(trap_cfg["rf_dc_electrodes"]),
        dc_bound_abs=dc_bound_abs,
        rf_dc_bound_abs=rf_dc_bound_abs,
        s_max=s_max,
    )
    bounded_objective = objective_label_to_solver_key(objective_label)

    bounded_kwargs = {
        "r0": r0,
        "freq_bounds": freq_bounds_hz,
        "principal_axis": principal_axis,
        "ref_dir": ref_dir,
        "alpha_deg": alpha_deg,
        "ion_mass_kg": constants.ion_mass,
        "ion_charge_c": constants.ion_charge,
        "poly_is_potential_energy": False,
        "freqs_in_hz": True,
        "trap_name": str(trap_cfg["trap_name"]),
        "dc_electrodes": list(trap_cfg["dc_electrodes"]),
        "rf_dc_electrodes": list(trap_cfg["rf_dc_electrodes"]),
        "num_samples": APP_MODEL_NUM_SAMPLES,
        "polyfit_deg": APP_POLYFIT_DEG,
        "use_cache": True,
    }

    with st.spinner("Running bounded inverse solve..."):
        bounded_out = solve_u_for_frequency_box(
            **bounded_kwargs,
            objective=bounded_objective,
            enforce_bounds=True,
            u_bounds=u_bounds,
        )

    bounded_bundle = make_inverse_result_bundle(
        label="Bounded result",
        trap_cfg=trap_cfg,
        solve_out=bounded_out,
        objective_label=objective_label,
        ref_rf_freq_hz=ref_rf_freq_hz,
        bounds_note=(
            f"|DC| <= {dc_bound_abs:g}, |RF-DC| <= {rf_dc_bound_abs:g}, "
            f"0 <= s <= {s_max:.6g}"
        ),
    )

    fallback_bundle = None
    if bounded_out.get("u") is None or bounded_out.get("status") != "ok":
        fallback_objective = (
            "l2" if bounded_objective in ("l2_dc", "s_min", "s_max") else bounded_objective
        )
        fallback_label = (
            "Fallback unconstrained best-effort (L2 min for all)"
            if fallback_objective != bounded_objective
            else f"Fallback unconstrained best-effort ({objective_label})"
        )
        with st.spinner("Running unconstrained fallback solve..."):
            fallback_out = solve_u_for_frequency_box(
                **bounded_kwargs,
                objective=fallback_objective,
                enforce_bounds=False,
                u_bounds=None,
            )
        fallback_bundle = make_inverse_result_bundle(
            label=fallback_label,
            trap_cfg=trap_cfg,
            solve_out=fallback_out,
            objective_label="L2 min for all" if fallback_objective == "l2" else objective_label,
            ref_rf_freq_hz=ref_rf_freq_hz,
            bounds_note="No control bounds applied.",
        )

    return {
        "trap_label": trap_label,
        "trap_name": trap_cfg["trap_name"],
        "bounded": bounded_bundle,
        "fallback": fallback_bundle,
        "requested_freq_bounds_hz": freq_bounds_hz,
        "requested_freq_bounds_mhz": [(lo / 1.0e6, hi / 1.0e6) for lo, hi in freq_bounds_hz],
    }


def make_inverse_result_bundle(
    *,
    label: str,
    trap_cfg: Mapping[str, Any],
    solve_out: Mapping[str, Any],
    objective_label: str,
    ref_rf_freq_hz: float,
    bounds_note: str,
) -> dict[str, Any]:
    bundle = {
        "label": label,
        "objective_label": objective_label,
        "status": solve_out.get("status"),
        "solve_out": solve_out,
        "bounds_note": bounds_note,
        "verification": None,
        "verification_error": None,
        "trap_cfg": dict(trap_cfg),
        "ref_rf_freq_hz": ref_rf_freq_hz,
    }
    if solve_out.get("u") is None:
        return bundle
    try:
        bundle["verification"] = build_forward_verification(
            trap_cfg=trap_cfg,
            solve_out=solve_out,
            ref_rf_freq_hz=ref_rf_freq_hz,
        )
    except Exception as exc:
        bundle["verification_error"] = str(exc)
    return bundle


def build_forward_verification(
    *,
    trap_cfg: Mapping[str, Any],
    solve_out: Mapping[str, Any],
    ref_rf_freq_hz: float,
) -> dict[str, Any]:
    u = np.asarray(solve_out["u"], dtype=float)
    dc_electrodes = list(trap_cfg["dc_electrodes"])
    rf_dc_electrodes = list(trap_cfg["rf_dc_electrodes"])
    s_val = float(solve_out["u_s"])
    rf_amp_v = convert_s_to_rf_amplitude(s_val, ref_rf_freq_hz)

    tv = Trapping_Vars()
    for electrode, value in zip(dc_electrodes, u[: len(dc_electrodes)]):
        tv.set_amp(tv.dc_key, electrode, float(value))
    rf_start = len(dc_electrodes)
    rf_stop = rf_start + len(rf_dc_electrodes)
    for electrode, value in zip(rf_dc_electrodes, u[rf_start:rf_stop]):
        tv.set_amp(tv.dc_key, electrode, float(value))
    tv.add_driving(
        "RF",
        float(ref_rf_freq_hz),
        0.0,
        {"RF1": rf_amp_v, "RF2": rf_amp_v},
    )

    sim = Simulation(str(trap_cfg["trap_name"]), Trapping_Vars())
    sim.change_electrode_variables(tv)
    sim.clear_held_results()
    sim.find_equilib_position_single(num_ions=1)
    sim.get_static_normal_modes_and_freq(1, normalize=True, sort_by_freq=True)
    sim.compute_principal_directions_from_one_ion()
    sim.populate_normalmodes_in_prinipledir_freq_labels()

    ppack = sim.principal_dir_normalmodes_andfrequencies.get(1)
    if ppack is None:
        raise RuntimeError("Principal-direction verification data was not populated.")

    eq_pos = np.asarray(sim.ion_equilibrium_positions.get(1), dtype=float)
    freqs_hz = np.asarray(ppack.get("frequencies_Hz"), dtype=float)
    principal_dirs = np.asarray(sim.principal_dirs, dtype=float)
    return {
        "equilibrium_position_m": eq_pos.reshape(-1, 3)[0],
        "frequencies_hz": freqs_hz.reshape(3),
        "principal_dirs": principal_dirs.reshape(3, 3),
        "rf_amp_v": rf_amp_v,
    }


def render_single_reachability_result(result: Mapping[str, Any] | None) -> None:
    if not result:
        return
    st.markdown(
        f"Trap: `{result['trap_label']}`  |  Boundary points: `{result['n_boundary_points']}`  |  "
        f"Cache hit: `{result['cache_hit']}`"
    )
    st.caption(
        f"Reference RF: {result['ref_rf_freq_hz'] / 1.0e6:.6f} MHz, "
        f"max RF amplitude: {result['max_rf_amp_v']:.6g} V, "
        f"max s: {result['s_max']:.6g}"
    )
    st.plotly_chart(result["frequency_figure"], use_container_width=True)
    if result.get("show_lambda") and result.get("lambda_figure") is not None:
        st.plotly_chart(result["lambda_figure"], use_container_width=True)


def render_multi_reachability_result(result: Mapping[str, Any] | None) -> None:
    if not result:
        return
    st.caption(
        f"Overlayed `{result['n_specs']}` specs with `{result['n_samples']}` reachability samples each."
    )
    st.plotly_chart(result["figure"], use_container_width=True)


def render_inverse_result(result: Mapping[str, Any] | None) -> None:
    if not result:
        return

    bounded = result["bounded"]
    bounded_status = str(bounded["status"])
    if bounded_status == "ok":
        st.success("Bounded request is feasible.")
    else:
        st.error(f"Bounded request is infeasible or failed: {bounded_status}")

    render_inverse_bundle(bounded)

    fallback = result.get("fallback")
    if fallback is not None:
        st.warning("Displaying fallback unconstrained best-effort result.")
        render_inverse_bundle(fallback)


def render_inverse_bundle(bundle: Mapping[str, Any]) -> None:
    solve_out = bundle["solve_out"]
    st.subheader(str(bundle["label"]))
    st.caption(f"Objective: {bundle['objective_label']}  |  {bundle['bounds_note']}")

    if solve_out.get("u") is None:
        solver_message = str(solve_out.get("solver_info", {}).get("message", "No solution vector returned."))
        st.write(solver_message)
        render_solver_diagnostics(solve_out)
        return

    u = np.asarray(solve_out["u"], dtype=float)
    trap_cfg = bundle["trap_cfg"]
    verification = bundle.get("verification")
    verification_error = bundle.get("verification_error")
    s_val = float(solve_out["u_s"])
    ref_rf_freq_hz = float(bundle["ref_rf_freq_hz"])
    rf_amp_v = (
        float(verification["rf_amp_v"])
        if verification is not None
        else convert_s_to_rf_amplitude(s_val, ref_rf_freq_hz)
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("s", format_numeric(s_val))
    metric_cols[1].metric("V_amp at f_ref", f"{rf_amp_v:.6g} V")
    metric_cols[2].metric("Reference RF frequency", f"{ref_rf_freq_hz / 1.0e6:.6f} MHz")

    render_control_layout(
        dc_electrodes=list(trap_cfg["dc_electrodes"]),
        dc_values=u[: len(trap_cfg["dc_electrodes"])],
        rf_dc_electrodes=list(trap_cfg["rf_dc_electrodes"]),
        rf_dc_values=solve_out["u_rf_dc"],
    )

    if verification_error:
        st.warning(f"Verification run failed: {verification_error}")
    elif verification is not None:
        render_verification_details(bundle=bundle, verification=verification)

    render_solver_diagnostics(solve_out)


def render_verification_details(*, bundle: Mapping[str, Any], verification: Mapping[str, Any]) -> None:
    solve_out = bundle["solve_out"]
    freq_df = pd.DataFrame(
        {
            "mode": ["f1", "f2", "f3"],
            "recovered_MHz": np.asarray(verification["frequencies_hz"], dtype=float) / 1.0e6,
            "recovered_Hz": np.asarray(verification["frequencies_hz"], dtype=float),
        }
    )
    principal_df = pd.DataFrame(
        np.asarray(verification["principal_dirs"], dtype=float),
        index=["dir_1", "dir_2", "dir_3"],
        columns=["x", "y", "z"],
    )
    summary_df = pd.DataFrame(
        [
            ("Status", solve_out.get("status")),
            ("Residual norm", solve_out.get("residual_norm")),
            ("u L2 norm", solve_out.get("u_norm2")),
            ("u L-inf norm", solve_out.get("u_norminf")),
            ("Cache hit", solve_out.get("cache_hit")),
            ("Cache path", solve_out.get("cache_path")),
            ("Equilibrium x (m)", verification["equilibrium_position_m"][0]),
            ("Equilibrium y (m)", verification["equilibrium_position_m"][1]),
            ("Equilibrium z (m)", verification["equilibrium_position_m"][2]),
        ],
        columns=["Field", "Value"],
    )

    with st.expander("Verification details", expanded=True):
        freq_col, summary_col = st.columns([1.1, 1.2])
        freq_col.dataframe(freq_df, use_container_width=True, hide_index=True)
        summary_col.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.dataframe(principal_df, use_container_width=True)


def render_solver_diagnostics(solve_out: Mapping[str, Any]) -> None:
    solver_info = solve_out.get("solver_info") or {}
    if not solver_info:
        return
    diag_df = pd.DataFrame(
        {
            "Field": list(solver_info.keys()),
            "Value": [solver_info[key] for key in solver_info.keys()],
        }
    )
    with st.expander("Solver diagnostics", expanded=False):
        st.dataframe(diag_df, use_container_width=True, hide_index=True)


def render_control_layout(
    *,
    dc_electrodes: Sequence[str],
    dc_values: Sequence[float],
    rf_dc_electrodes: Sequence[str],
    rf_dc_values: Sequence[float],
) -> None:
    st.markdown(CONTROL_GRID_STYLE, unsafe_allow_html=True)
    render_named_grid("DC controls", dc_layout_rows(dc_electrodes, dc_values))
    render_named_grid("RF-DC controls", [list(zip(rf_dc_electrodes, rf_dc_values))])


def render_named_grid(title: str, rows: Sequence[Sequence[tuple[str, float] | None]]) -> None:
    if not rows:
        return
    n_cols = max(len(row) for row in rows)
    html_parts = [f"<div class='inverse-app-grid-title'>{title}</div>"]
    for row in rows:
        html_parts.append(f"<div class='inverse-app-grid cols-{n_cols}'>")
        for cell in row:
            if cell is None:
                html_parts.append("<div class='inverse-app-cell empty'></div>")
                continue
            name, value = cell
            html_parts.append(
                "<div class='inverse-app-cell'>"
                f"<div class='inverse-app-name'>{name}</div>"
                f"<div class='inverse-app-value'>{format_numeric(value)} V</div>"
                "</div>"
            )
        html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def dc_layout_rows(
    dc_electrodes: Sequence[str],
    dc_values: Sequence[float],
) -> list[list[tuple[str, float] | None]]:
    cells = list(zip(dc_electrodes, np.asarray(dc_values, dtype=float).tolist()))
    n_dc = len(cells)
    if n_dc == 10:
        return [cells[0:5], cells[5:10]]
    if n_dc == 12:
        return [cells[0:3] + [None] + cells[3:6], cells[6:9] + [None] + cells[9:12]]
    if n_dc == 20:
        return [cells[0:5], cells[5:10], cells[10:15], cells[15:20]]
    return [cells[idx : idx + 5] for idx in range(0, n_dc, 5)]


def get_trap_registry_entry(display_label: str) -> Mapping[str, Any]:
    if display_label not in constants.INVERSE_APP_TRAP_REGISTRY:
        raise ValueError(f"Unknown trap selection: {display_label}")
    return constants.INVERSE_APP_TRAP_REGISTRY[display_label]


def build_u_bounds(
    *,
    n_dc: int,
    n_rf: int,
    dc_bound_abs: float,
    rf_dc_bound_abs: float,
    s_max: float,
) -> list[tuple[float, float]]:
    return [(-dc_bound_abs, dc_bound_abs)] * n_dc + [(-rf_dc_bound_abs, rf_dc_bound_abs)] * n_rf + [
        (0.0, s_max)
    ]


def compute_s_max(*, ref_rf_freq_hz: float, max_rf_amp_v: float) -> float:
    if ref_rf_freq_hz <= 0.0:
        raise ValueError("Reference RF frequency must be positive.")
    if max_rf_amp_v < 0.0:
        raise ValueError("Max RF amplitude must be nonnegative.")
    omega_ref_mhz = 2.0 * math.pi * (ref_rf_freq_hz / 1.0e6)
    if omega_ref_mhz <= 0.0:
        raise ValueError("Reference RF frequency produced a nonpositive omega.")
    return float((max_rf_amp_v**2) / (omega_ref_mhz**2))


def convert_s_to_rf_amplitude(s_val: float, ref_rf_freq_hz: float) -> float:
    omega_ref_mhz = 2.0 * math.pi * (ref_rf_freq_hz / 1.0e6)
    return float(math.sqrt(max(s_val, 0.0)) * omega_ref_mhz)


def parse_principal_direction_inputs(
    principal_axis_text: str,
    ref_dir_text: str,
    alpha_deg_text: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    principal_axis = parse_vector_text(principal_axis_text, "principal_axis")
    ref_dir = parse_vector_text(ref_dir_text, "ref_dir")
    alpha_deg = parse_required_float(alpha_deg_text, "alpha_deg")
    rotation_from_axis_ref_alpha(principal_axis, ref_dir, alpha_deg)
    return principal_axis, ref_dir, alpha_deg


def parse_vector_text(value: str, field_name: str) -> np.ndarray:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        cleaned = text.strip("[]()")
        parts = [part for part in re.split(r"[\s,]+", cleaned) if part]
        if len(parts) != 3:
            raise ValueError(f"{field_name} must be a 3-vector such as [1, 0, 0].") from None
        try:
            parsed = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"{field_name} must contain numeric entries.") from exc

    arr = np.asarray(parsed, dtype=float).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"{field_name} must contain exactly 3 numbers.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} must contain only finite numbers.")
    return arr


def parse_required_float(
    value: Any,
    field_name: str,
    *,
    min_value: float | None = None,
) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{field_name} must be >= {min_value:g}.")
    return parsed


def parse_frequency_bounds_mhz(
    bounds_text: Sequence[tuple[str, str]],
) -> list[tuple[float, float]]:
    if len(bounds_text) != 3:
        raise ValueError("Exactly three mode bounds are required.")
    out: list[tuple[float, float]] = []
    for idx, (lo_text, hi_text) in enumerate(bounds_text, start=1):
        lo = parse_required_float(lo_text, f"Mode {idx} lower bound", min_value=0.0)
        hi = parse_required_float(hi_text, f"Mode {idx} upper bound", min_value=0.0)
        if lo > hi:
            raise ValueError(f"Mode {idx} lower bound must be <= upper bound.")
        out.append((lo * 1.0e6, hi * 1.0e6))
    return out


def parse_multi_spec_text(raw_specs_text: str) -> list[str | Mapping[str, Any]]:
    text = str(raw_specs_text).strip()
    if not text:
        raise ValueError("The multi-spec textbox is empty.")

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        helper_scope = {
            "RF_S_MAX_DEFAULT": constants.RF_S_MAX_DEFAULT,
            "range": range,
        }
        try:
            parsed = eval(text, {"__builtins__": {}}, helper_scope)
        except Exception as exc:
            raise ValueError(f"Could not parse multi-spec textbox: {exc}") from exc

    if isinstance(parsed, Mapping):
        items: list[str | Mapping[str, Any]] = [parsed]
    elif isinstance(parsed, (list, tuple)):
        items = list(parsed)
    else:
        raise ValueError("Multi-spec textbox must evaluate to a dict, list, or tuple.")

    normalized: list[str | Mapping[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping):
            normalized.append(apply_registry_aliases(dict(item)))
        elif isinstance(item, str):
            normalized.append(item)
        else:
            raise ValueError("Each multi-spec entry must be either a trap name string or a mapping.")
    return normalized


def apply_registry_aliases(spec: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(spec)
    candidates = []
    for key in ("display_label", "trap_label", "trap_name", "name"):
        value = cfg.get(key)
        if isinstance(value, str):
            candidates.append(value)

    for candidate in candidates:
        registry_entry = constants.INVERSE_APP_TRAP_REGISTRY.get(candidate)
        if registry_entry is None:
            continue
        cfg.setdefault("display_label", registry_entry["display_label"])
        cfg.setdefault("name", registry_entry["display_label"])
        if cfg.get("trap_name") in (None, candidate):
            cfg["trap_name"] = registry_entry["trap_name"]
        cfg.setdefault("dc_electrodes", list(registry_entry["dc_electrodes"]))
        cfg.setdefault("rf_dc_electrodes", list(registry_entry["rf_dc_electrodes"]))
        break
    return cfg


def objective_label_to_solver_key(label: str) -> str:
    if label == "L2 min for all":
        return "l2"
    if label == "L2 min for all but s":
        return "l2_dc"
    if label == "L2_Sx100":
        return "l2_sx100"
    if label == "s_min":
        return "s_min"
    if label == "s_max":
        return "s_max"
    raise ValueError(f"Unsupported minimization routine: {label}")


def format_numeric(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(number)
    if number == 0.0:
        return "0"
    if abs(number) >= 1.0e4 or abs(number) < 1.0e-3:
        return f"{number:+.4e}"
    return f"{number:+.6f}"


if __name__ == "__main__":
    main()
