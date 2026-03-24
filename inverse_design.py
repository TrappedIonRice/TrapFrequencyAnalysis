"""
Inverse design entrypoint: map (r0, freqs, principal directions) to controls u.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import constants

from control_constraints import (
    build_L_b_for_point,
    build_modal_constraints_for_point,
    build_target_hessian,
)
from linearsyssolve101 import build_voltage_to_c_matrix, build_voltage_to_c_matrix_cached
from trap_A_cache import DEFAULT_CACHE_DIR

try:
    from scipy.optimize import linprog, minimize
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    linprog = None
    minimize = None
    _HAVE_SCIPY = False


def solve_u_for_targets(
    *,
    r0: np.ndarray,
    freqs: np.ndarray | None = None,
    freq_bounds: Sequence[Tuple[float | None, float | None]] | None = None,
    target_mode: str = "exact",
    principal_axis: np.ndarray,
    ref_dir: np.ndarray,
    alpha_deg: float,
    ion_mass_kg: float,
    ion_charge_c: float | None = None,
    poly_is_potential_energy: bool = True,
    freqs_in_hz: bool = True,
    trap_name: str,
    dc_electrodes: List[str],
    rf_dc_electrodes: List[str] = ("RF1", "RF2"),
    num_samples: int,
    dc_bounds: Tuple[float, float] | List[Tuple[float, float]] = (-500.0, 500.0),
    rf_dc_bounds: Tuple[float, float] | List[Tuple[float, float]] = (-50.0, 50.0),
    s_bounds: Tuple[float, float] = (0.0, constants.RF_S_MAX_DEFAULT),
    polyfit_deg: int = 4,
    seed: int = 0,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_rebuild_A: bool = False,
    objective: str = "l2",
    enforce_bounds: bool = False,
    u_bounds: List[Tuple[float, float]] | None = None,
    ridge_lambda: float = 0.0,
    s_penalty_scale: float = 1e-5,
    G_ub: np.ndarray | None = None,
    h_ub: np.ndarray | None = None,
) -> Dict[str, object]:
    """
    Solve for control vector u under principal-basis frequency constraints.

    Modes:
      - target_mode='exact': use freqs as exact targets (legacy behavior).
      - target_mode='box': use freq_bounds as per-mode bounds.
    """
    r0v = np.asarray(r0, dtype=float).reshape(-1)
    if r0v.shape[0] != 3:
        raise ValueError("r0 must have shape (3,)")
    mode = str(target_mode).strip().lower()
    freq_bounds_use = _resolve_freq_bounds_request(
        target_mode=mode,
        freqs=freqs,
        freq_bounds=freq_bounds,
    )

    # Build linear map c = A @ u and powers
    if use_cache:
        out = build_voltage_to_c_matrix_cached(
            trap_name=trap_name,
            dc_electrodes=dc_electrodes,
            rf_dc_electrodes=rf_dc_electrodes,
            num_samples=num_samples,
            dc_bounds=dc_bounds,
            rf_dc_bounds=rf_dc_bounds,
            s_bounds=s_bounds,
            polyfit_deg=polyfit_deg,
            seed=seed,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild_A,
            ion_mass_kg=ion_mass_kg,
        )
    else:
        out = build_voltage_to_c_matrix(
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
    A = np.asarray(out["A"], dtype=float)
    powers = np.asarray(out["powers"], dtype=int)

    if mode == "exact":
        # Backward-compatibility: exact mode keeps legacy Cartesian Hessian rows
        # in returned L,b,M (build_target_hessian + build_L_b_for_point path).
        freqs_v = np.asarray(freqs, dtype=float).reshape(-1)
        Kstar = build_target_hessian(
            freqs_v,
            principal_axis,
            ref_dir,
            alpha_deg,
            mass=ion_mass_kg,
            charge=ion_charge_c,
            poly_is_potential_energy=poly_is_potential_energy,
            freqs_in_hz=freqs_in_hz,
        )
        L, b = build_L_b_for_point(
            powers,
            r0v,
            Kstar,
            include_gradient=True,
            basis="nondim",
            nd_L0_m=constants.ND_L0_M,
        )
        Mmat = L @ A
        G_modal_u = None
        h_modal_u = None
    else:
        # Box mode uses modal-basis equalities and inequalities.
        L_eq_c, b_eq, G_ub_c, h_ub_c = build_modal_constraints_for_point(
            powers,
            r0v,
            principal_axis=principal_axis,
            ref_dir=ref_dir,
            alpha_deg=alpha_deg,
            freq_bounds_hz_or_rad=freq_bounds_use,
            mass=ion_mass_kg,
            charge=ion_charge_c,
            poly_is_potential_energy=poly_is_potential_energy,
            freqs_in_hz=freqs_in_hz,
            include_gradient=True,
            basis="nondim",
            nd_L0_m=constants.ND_L0_M,
        )
        L = L_eq_c
        b = b_eq
        Mmat = L @ A
        G_modal_u = G_ub_c @ A if G_ub_c.shape[0] > 0 else None
        h_modal_u = h_ub_c.copy() if G_ub_c.shape[0] > 0 else None

    Gmat: np.ndarray | None = G_modal_u
    hvec: np.ndarray | None = h_modal_u
    if G_ub is not None or h_ub is not None:
        if G_ub is None or h_ub is None:
            raise ValueError("G_ub and h_ub must be provided together")
        G_extra = np.asarray(G_ub, dtype=float)
        h_extra = np.asarray(h_ub, dtype=float).reshape(-1)
        if G_extra.ndim != 2 or G_extra.shape[1] != Mmat.shape[1]:
            raise ValueError("G_ub must have shape (n_ineq, n_controls)")
        if G_extra.shape[0] != h_extra.shape[0]:
            raise ValueError("h_ub length must equal number of G_ub rows")
        if G_extra.shape[0] > 0:
            if Gmat is None:
                Gmat = G_extra
                hvec = h_extra
            else:
                Gmat = np.vstack([Gmat, G_extra])
                hvec = np.concatenate([hvec, h_extra])

    # Solve for u
    solver_info: Dict[str, object] = {}
    status = "ok"
    if objective not in ("l2", "linf", "weighted_l2", "avg_max_dc", "l2_dc", "l2_sx100", "s_min", "s_max"):
        raise ValueError(
            "objective must be 'l2', 'linf', 'weighted_l2', 'avg_max_dc', 'l2_dc', 'l2_sx100', 's_min', or 's_max'"
        )

    has_ineq = Gmat is not None and hvec is not None and Gmat.shape[0] > 0
    bounded_mode = (
        enforce_bounds
        or (u_bounds is not None)
        or (objective in ("linf", "avg_max_dc", "l2_dc"))
        or has_ineq
    )

    # Unbounded best-effort diagnostic
    u_ls, resid_ls = _lsq_best_effort(Mmat, b)
    eq_tol = _eq_tol(b)
    ineq_tol = _ineq_tol(hvec)
    ineq_violation_ls = _ineq_violation(Mmat=Gmat, u=u_ls, b=hvec)
    ineq_violation_rel_ls = _ineq_violation_relative(Mmat=Gmat, u=u_ls, b=hvec)
    solver_info.update(
        {
            "resid_ls": resid_ls,
            "eq_tol": eq_tol,
            "ineq_tol": ineq_tol,
            "ineq_violation_ls": ineq_violation_ls,
            "ineq_violation_rel_ls": ineq_violation_rel_ls,
            "has_ineq_constraints": has_ineq,
            "target_feasible_unbounded": (resid_ls <= eq_tol) and (ineq_violation_ls <= ineq_tol),
        }
    )
    if objective == "l2":
        if bounded_mode:
            if not _HAVE_SCIPY:
                raise RuntimeError("SciPy is required for constrained l2 solve")
            bounds_arg = u_bounds
            if bounds_arg is None and enforce_bounds:
                bounds_arg = _build_u_bounds_from_blocks(
                    dc_bounds, rf_dc_bounds, s_bounds, len(dc_electrodes), len(rf_dc_electrodes)
                )
            u, info = _solve_l2_constrained(Mmat, b, bounds_arg, G_ub=Gmat, h_ub=hvec)
            solver_info.update(info)
            solver_info["solver_name"] = "l2_constrained"
            solver_info["message"] = info.get("message", "")
        else:
            u, info = _solve_l2_min_norm(Mmat, b, ridge_lambda=ridge_lambda)
            solver_info.update(info)
            solver_info["solver_name"] = "l2_min_norm"
            solver_info["message"] = info.get("message", "")
    elif objective in ("weighted_l2", "l2_sx100"):
        s_penalty_scale_use = 100.0 if objective == "l2_sx100" else s_penalty_scale
        if bounded_mode:
            if not _HAVE_SCIPY:
                raise RuntimeError("SciPy is required for constrained weighted_l2 solve")
            bounds_arg = u_bounds
            if bounds_arg is None and enforce_bounds:
                bounds_arg = _build_u_bounds_from_blocks(
                    dc_bounds, rf_dc_bounds, s_bounds, len(dc_electrodes), len(rf_dc_electrodes)
                )
            u, info = _solve_weighted_l2_constrained(
                Mmat,
                b,
                bounds_arg,
                s_penalty_scale_use,
                G_ub=Gmat,
                h_ub=hvec,
            )
            solver_info.update(info)
            solver_info["solver_name"] = (
                "l2_sx100_constrained" if objective == "l2_sx100" else "weighted_l2_constrained"
            )
            solver_info["message"] = info.get("message", "")
        else:
            u, info = _solve_weighted_l2_min_norm(
                Mmat, b, s_penalty_scale=s_penalty_scale_use, ridge_lambda=ridge_lambda
            )
            solver_info.update(info)
            solver_info["solver_name"] = (
                "l2_sx100_min_norm" if objective == "l2_sx100" else "weighted_l2_min_norm"
            )
            solver_info["message"] = info.get("message", "")
    elif objective == "l2_dc":
        if u_bounds is None:
            raise ValueError("u_bounds is required for objective='l2_dc'")
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy is required for l2_dc solve")
        bounds_clean = _normalize_bounds_allow_nan(u_bounds, Mmat.shape[1])
        lo, hi = bounds_clean[-1]
        if lo is None and hi is None:
            raise ValueError("u_bounds must include at least one bound for s (last entry)")
        weights = np.ones(Mmat.shape[1], dtype=float)
        weights[-1] = 0.0  # do not penalize s in objective
        u, info = _solve_weighted_l2_constrained_custom(
            Mmat,
            b,
            bounds_clean,
            weights,
            G_ub=Gmat,
            h_ub=hvec,
        )
        solver_info.update(info)
        solver_info["solver_name"] = "l2_dc_constrained"
        solver_info["message"] = info.get("message", "")
    elif objective == "avg_max_dc":
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy is required for avg_max_dc solve")
        K = len(dc_electrodes)
        K_rf = len(rf_dc_electrodes)
        dc_indices = list(range(0, K))
        rf_dc_indices = list(range(K, K + K_rf))
        if u_bounds is None:
            u_bounds = _build_u_bounds_from_blocks(dc_bounds, rf_dc_bounds, s_bounds, K, K_rf)
        u, info = _solve_avg_max_dc_lp(
            Mmat,
            b,
            dc_indices,
            rf_dc_indices,
            u_bounds,
            G_ub=Gmat,
            h_ub=hvec,
        )
        solver_info.update(info)
        solver_info["solver_name"] = "avg_max_dc_lp"
        solver_info["message"] = info.get("message", "")
    elif objective in ("s_min", "s_max"):
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy is required for s_min/s_max solve")
        bounds_arg = u_bounds
        if bounds_arg is None and enforce_bounds:
            bounds_arg = _build_u_bounds_from_blocks(
                dc_bounds, rf_dc_bounds, s_bounds, len(dc_electrodes), len(rf_dc_electrodes)
            )
        if bounds_arg is None:
            raise ValueError(f"u_bounds is required for objective='{objective}'")
        u, info = _solve_s_coordinate_constrained(
            Mmat,
            b,
            bounds_arg,
            maximize_s=(objective == "s_max"),
            G_ub=Gmat,
            h_ub=hvec,
        )
        solver_info.update(info)
        solver_info["solver_name"] = f"{objective}_constrained"
        solver_info["message"] = info.get("message", "")
    else:
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy is required for linf solve")
        u, info = _solve_linf_lp(Mmat, b, u_bounds, G_ub=Gmat, h_ub=hvec)
        solver_info.update(info)
        solver_info["solver_name"] = "linf_lp"
        solver_info["message"] = info.get("message", "")

    # Post-solve feasibility checks
    if u is None:
        if bounded_mode:
            msg = (
                f"Not feasible given bounds. resid_min={resid_ls:.3e}, "
                f"eq_tol={eq_tol:.3e}, ineq_violation_ls={ineq_violation_ls:.3e}, "
                f"ineq_violation_rel_ls={ineq_violation_rel_ls:.3e}, "
                f"ineq_tol={ineq_tol:.3e}, solver={solver_info.get('solver_name','')}"
            )
            # print(msg + " Try loosening bounds or changing target.")
            solver_info["message"] = msg
            solver_info["resid"] = resid_ls
            solver_info["ineq_violation"] = ineq_violation_ls
            solver_info["ineq_violation_rel"] = ineq_violation_rel_ls
            status = "infeasible_bounds" if _solver_reports_infeasible(solver_info) else "solver_failed"
            return _pack_result(
                u=None,
                status=status,
                solver_info=solver_info,
                A=A,
                powers=powers,
                L=L,
                b=b,
                Mmat=Mmat,
                cache_out=out,
            )
        # Unbounded mode: fall back to best-effort LS solution
        u = u_ls

    resid = _eq_residual(Mmat, u, b)
    ineq_violation = _ineq_violation(Mmat=Gmat, u=u, b=hvec)
    ineq_violation_rel = _ineq_violation_relative(Mmat=Gmat, u=u, b=hvec)
    solver_info["resid"] = resid
    solver_info["eq_tol"] = eq_tol
    solver_info["ineq_violation"] = ineq_violation
    solver_info["ineq_violation_rel"] = ineq_violation_rel
    solver_info["ineq_tol"] = ineq_tol

    if bounded_mode:
        if resid <= eq_tol and ineq_violation <= ineq_tol:
            status = "ok"
        else:
            msg = (
                f"Solver returned but constraints not met within tolerance. "
                f"resid={resid:.3e}, eq_tol={eq_tol:.3e}, "
                f"ineq_violation={ineq_violation:.3e}, ineq_tol={ineq_tol:.3e}, "
                f"ineq_violation_rel={ineq_violation_rel:.3e}, "
                f"solver={solver_info.get('solver_name','')}"
            )
            print(msg + " Try loosening bounds or changing target.")
            solver_info["message"] = msg
            status = "solver_failed"
            return _pack_result(
                u=None,
                status=status,
                solver_info=solver_info,
                A=A,
                powers=powers,
                L=L,
                b=b,
                Mmat=Mmat,
                cache_out=out,
            )
    else:
        if resid <= eq_tol:
            status = "ok"
        else:
            msg = (
                "Target not exactly feasible (b not in range(M) within tol). "
                "Returning least-squares best-effort."
            )
            solver_info["message"] = msg
            status = "best_effort_infeasible"
            u = u_ls
            resid = resid_ls
            solver_info["resid"] = resid

    u_norm2 = float(np.linalg.norm(u))
    u_norminf = float(np.max(np.abs(u)))
    s_val = float(u[-1])
    if s_val < 0 and status == "ok" and not bounded_mode:
        status = "best_effort_infeasible"
        solver_info["message"] = (
            solver_info.get("message", "")
            + " s is negative; consider bounds if s>=0 is required."
        ).strip()

    return {
        "u": u,
        "u_dc": u[: len(dc_electrodes)],
        "u_rf_dc": u[len(dc_electrodes) : len(dc_electrodes) + len(rf_dc_electrodes)],
        "u_s": s_val,
        "A": A,
        "powers": powers,
        "L": L,
        "b": b,
        "M": Mmat,
        "G_ub": Gmat,
        "h_ub": hvec,
        "residual_norm": resid,
        "u_norm2": u_norm2,
        "u_norminf": u_norminf,
        "status": status,
        "solver_info": solver_info,
        "cache_hit": out.get("cache_hit", False),
        "cache_path": out.get("cache_path"),
        "cfg": out.get("cfg"),
    }


def solve_u_for_exact_targets(
    *,
    freqs: np.ndarray,
    **kwargs,
) -> Dict[str, object]:
    """
    Convenience wrapper for exact-frequency inverse solve.
    """
    return solve_u_for_targets(
        freqs=freqs,
        freq_bounds=None,
        target_mode="exact",
        **kwargs,
    )


def solve_u_for_frequency_box(
    *,
    freq_bounds: Sequence[Tuple[float | None, float | None]],
    **kwargs,
) -> Dict[str, object]:
    """
    Convenience wrapper for per-mode frequency-box inverse solve.
    """
    return solve_u_for_targets(
        freqs=None,
        freq_bounds=freq_bounds,
        target_mode="box",
        **kwargs,
    )


def _resolve_freq_bounds_request(
    *,
    target_mode: str,
    freqs: np.ndarray | None,
    freq_bounds: Sequence[Tuple[float | None, float | None]] | None,
) -> List[Tuple[float | None, float | None]]:
    mode = str(target_mode).strip().lower()
    if mode == "exact":
        if freq_bounds is not None:
            raise ValueError("freq_bounds must be omitted when target_mode='exact'")
        if freqs is None:
            raise ValueError("freqs is required when target_mode='exact'")
        freqs_v = np.asarray(freqs, dtype=float).reshape(-1)
        if freqs_v.shape[0] != 3:
            raise ValueError("freqs must have shape (3,)")
        if np.any(~np.isfinite(freqs_v)) or np.any(freqs_v < 0.0):
            raise ValueError("freqs entries must be finite and >= 0")
        return [(float(f), float(f)) for f in freqs_v]

    if mode == "box":
        if freqs is not None:
            raise ValueError("freqs must be omitted when target_mode='box'")
        return _normalize_frequency_bounds_input(freq_bounds)

    raise ValueError("target_mode must be 'exact' or 'box'")


def _normalize_frequency_bounds_input(
    freq_bounds: Sequence[Tuple[float | None, float | None]] | None,
) -> List[Tuple[float | None, float | None]]:
    if freq_bounds is None:
        raise ValueError("freq_bounds is required when target_mode='box'")
    if len(freq_bounds) != 3:
        raise ValueError("freq_bounds must have length 3")
    out: List[Tuple[float | None, float | None]] = []
    for i, pair in enumerate(freq_bounds):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"freq_bounds[{i}] must be a (lower, upper) pair")
        lo = _normalize_frequency_bound_value(pair[0], is_lower=True, mode_index=i)
        hi = _normalize_frequency_bound_value(pair[1], is_lower=False, mode_index=i)
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"freq_bounds lower > upper at mode index {i}")
        out.append((lo, hi))
    return out


def _normalize_frequency_bound_value(
    value: float | None,
    *,
    is_lower: bool,
    mode_index: int,
) -> float | None:
    if value is None:
        return None
    v = float(value)
    if np.isnan(v):
        raise ValueError(f"freq_bounds cannot contain NaN (mode index {mode_index})")
    if np.isinf(v):
        if is_lower:
            if v > 0:
                raise ValueError(f"lower frequency bound cannot be +inf at mode index {mode_index}")
            return None
        if v < 0:
            raise ValueError(f"upper frequency bound cannot be -inf at mode index {mode_index}")
        return None
    if v < 0.0:
        raise ValueError(f"frequency bounds must be >= 0 (mode index {mode_index})")
    return v


def _bounds_needed_for_s(u_bounds: List[Tuple[float, float]] | None) -> bool:
    if u_bounds is None:
        return False
    if len(u_bounds) == 0:
        return True
    lo, _ = u_bounds[-1]
    if lo is None:
        return True
    if isinstance(lo, float) and np.isnan(lo):
        return True
    return lo < 0.0


def _eq_tol(b: np.ndarray) -> float:
    return 1e-9 * (1.0 + float(np.linalg.norm(b)))


def _eq_residual(M: np.ndarray, u: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(M @ u - b))


def _ineq_tol(h: np.ndarray | None) -> float:
    """
    Scale-aware inequality tolerance for constraints G u <= h.
    Mirrors _eq_tol style by scaling with RHS magnitude.
    """
    if h is None or np.asarray(h).size == 0:
        return 0.0
    hv = np.asarray(h, dtype=float).reshape(-1)
    return 1e-9 * (1.0 + float(np.linalg.norm(hv)))


def _ineq_violation(
    *,
    Mmat: np.ndarray | None,
    u: np.ndarray,
    b: np.ndarray | None,
) -> float:
    if Mmat is None or b is None:
        return 0.0
    if Mmat.size == 0:
        return 0.0
    viol = np.asarray(Mmat, dtype=float) @ np.asarray(u, dtype=float) - np.asarray(b, dtype=float)
    if viol.size == 0:
        return 0.0
    vmax = float(np.max(viol))
    return 0.0 if vmax <= 0.0 else vmax


def _ineq_violation_relative(
    *,
    Mmat: np.ndarray | None,
    u: np.ndarray,
    b: np.ndarray | None,
) -> float:
    """
    Dimensionless inequality violation:
      max_i ((G u - h)_i / (1 + |h_i|), 0)
    """
    if Mmat is None or b is None:
        return 0.0
    if Mmat.size == 0:
        return 0.0
    hv = np.asarray(b, dtype=float).reshape(-1)
    viol = np.asarray(Mmat, dtype=float) @ np.asarray(u, dtype=float) - hv
    if viol.size == 0:
        return 0.0
    scale = 1.0 + np.abs(hv)
    vmax = float(np.max(viol / scale))
    return 0.0 if vmax <= 0.0 else vmax


def _lsq_best_effort(M: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    u_ls, *_ = np.linalg.lstsq(M, b, rcond=None)
    resid = _eq_residual(M, u_ls, b)
    return u_ls, resid


def _solver_reports_infeasible(solver_info: Dict[str, object]) -> bool:
    msg = str(solver_info.get("message", "")).lower()
    if "infeasible" in msg:
        return True
    status = solver_info.get("status")
    if isinstance(status, int) and status in (2,):
        return True
    return False


def _pack_result(
    *,
    u: np.ndarray | None,
    status: str,
    solver_info: Dict[str, object],
    A: np.ndarray,
    powers: np.ndarray,
    L: np.ndarray,
    b: np.ndarray,
    Mmat: np.ndarray,
    cache_out: Dict[str, object],
) -> Dict[str, object]:
    return {
        "u": None if u is None else u,
        "u_dc": None,
        "u_rf_dc": None,
        "u_s": None,
        "A": A,
        "powers": powers,
        "L": L,
        "b": b,
        "M": Mmat,
        "residual_norm": float("nan"),
        "u_norm2": None,
        "u_norminf": None,
        "status": status,
        "solver_info": solver_info,
        "cache_hit": cache_out.get("cache_hit", False),
        "cache_path": cache_out.get("cache_path"),
        "cfg": cache_out.get("cfg"),
    }


def _solve_l2_min_norm(
    Mmat: np.ndarray, b: np.ndarray, *, ridge_lambda: float = 0.0
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Solve min ||u||_2 s.t. M u = b (if ridge_lambda==0) or
    min ||M u - b||^2 + ridge_lambda ||u||^2 otherwise.
    """
    info: Dict[str, object] = {}
    M = np.asarray(Mmat, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    if ridge_lambda and ridge_lambda > 0.0:
        MtM = M.T @ M
        rhs = M.T @ b
        u = np.linalg.solve(MtM + ridge_lambda * np.eye(M.shape[1]), rhs)
        info["used_ridge"] = True
        info["ridge_lambda"] = ridge_lambda
        return u, info

    # Minimum norm solution to equality constraints
    MMt = M @ M.T
    try:
        x = np.linalg.solve(MMt, b)
        u = M.T @ x
        info["used_ridge"] = False
        return u, info
    except np.linalg.LinAlgError:
        # Fallback to ridge if singular
        lam = ridge_lambda if ridge_lambda > 0 else 1e-8
        x = np.linalg.solve(MMt + lam * np.eye(MMt.shape[0]), b)
        u = M.T @ x
        info["used_ridge"] = True
        info["ridge_lambda"] = lam
    return u, info


def _solve_weighted_l2_min_norm(
    Mmat: np.ndarray,
    b: np.ndarray,
    *,
    s_penalty_scale: float,
    ridge_lambda: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Solve min ||W u||_2 s.t. M u = b, where W is diagonal and s has lower weight.
    s_penalty_scale < 1 means s is penalized less than DC entries.
    """
    info: Dict[str, object] = {}
    M = np.asarray(Mmat, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = M.shape[1]

    w = np.ones(n, dtype=float)
    w[-1] = float(s_penalty_scale)
    if w[-1] <= 0.0:
        raise ValueError("s_penalty_scale must be positive")

    # v = W u, minimize ||v|| subject to M W^{-1} v = b
    Winv = np.diag(1.0 / w)
    Mtilde = M @ Winv

    if ridge_lambda and ridge_lambda > 0.0:
        MtM = Mtilde.T @ Mtilde
        rhs = Mtilde.T @ b
        v = np.linalg.solve(MtM + ridge_lambda * np.eye(n), rhs)
        u = Winv @ v
        info["used_ridge"] = True
        info["ridge_lambda"] = ridge_lambda
        info["s_penalty_scale"] = w[-1]
        return u, info

    MMt = Mtilde @ Mtilde.T
    try:
        x = np.linalg.solve(MMt, b)
        v = Mtilde.T @ x
        u = Winv @ v
        info["used_ridge"] = False
        info["s_penalty_scale"] = w[-1]
        return u, info
    except np.linalg.LinAlgError:
        lam = ridge_lambda if ridge_lambda > 0 else 1e-8
        x = np.linalg.solve(MMt + lam * np.eye(MMt.shape[0]), b)
        v = Mtilde.T @ x
        u = Winv @ v
        info["used_ridge"] = True
        info["ridge_lambda"] = lam
        info["s_penalty_scale"] = w[-1]
        return u, info


def _solve_l2_constrained(
    Mmat: np.ndarray,
    b: np.ndarray,
    u_bounds: List[Tuple[float, float]] | None,
    *,
    G_ub: np.ndarray | None = None,
    h_ub: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Constrained min 0.5||u||^2 s.t. M u = b and bounds (including s>=0).
    """
    M = np.asarray(Mmat, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = M.shape[1]

    bounds = _normalize_bounds(u_bounds, n)
    # Enforce s >= 0
    lo, hi = bounds[-1]
    if lo is None or lo < 0.0:
        bounds[-1] = (0.0, hi)

    def obj(u: np.ndarray) -> float:
        return 0.5 * float(u @ u)

    def jac(u: np.ndarray) -> np.ndarray:
        return u

    cons = [
        {
            "type": "eq",
            "fun": lambda u: M @ u - b,
            "jac": lambda u: M,
        }
    ]
    if G_ub is not None and h_ub is not None and np.asarray(G_ub).size > 0:
        Gv = np.asarray(G_ub, dtype=float)
        hv = np.asarray(h_ub, dtype=float).reshape(-1)
        cons.append(
            {
                "type": "ineq",
                "fun": lambda u, G=Gv, h=hv: h - G @ u,
                "jac": lambda u, G=Gv: -G,
            }
        )

    u0 = np.zeros(n, dtype=float)
    res = minimize(obj, u0, jac=jac, constraints=cons, bounds=bounds, method="SLSQP")
    info = {"success": bool(res.success), "message": res.message}
    if not res.success:
        return None, info
    return np.asarray(res.x, dtype=float), info


def _solve_weighted_l2_constrained(
    Mmat: np.ndarray,
    b: np.ndarray,
    u_bounds: List[Tuple[float, float]] | None,
    s_penalty_scale: float,
    *,
    G_ub: np.ndarray | None = None,
    h_ub: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Constrained min 0.5||W u||^2 s.t. M u = b and bounds (including s>=0).
    """
    M = np.asarray(Mmat, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = M.shape[1]

    w = np.ones(n, dtype=float)
    w[-1] = float(s_penalty_scale)
    if w[-1] <= 0.0:
        raise ValueError("s_penalty_scale must be positive")

    bounds = _normalize_bounds(u_bounds, n)
    lo, hi = bounds[-1]
    if lo is None or lo < 0.0:
        bounds[-1] = (0.0, hi)

    def obj(u: np.ndarray) -> float:
        return 0.5 * float((w * u) @ (w * u))

    def jac(u: np.ndarray) -> np.ndarray:
        return (w * w) * u

    cons = [
        {
            "type": "eq",
            "fun": lambda u: M @ u - b,
            "jac": lambda u: M,
        }
    ]
    if G_ub is not None and h_ub is not None and np.asarray(G_ub).size > 0:
        Gv = np.asarray(G_ub, dtype=float)
        hv = np.asarray(h_ub, dtype=float).reshape(-1)
        cons.append(
            {
                "type": "ineq",
                "fun": lambda u, G=Gv, h=hv: h - G @ u,
                "jac": lambda u, G=Gv: -G,
            }
        )

    u0 = np.zeros(n, dtype=float)
    res = minimize(obj, u0, jac=jac, constraints=cons, bounds=bounds, method="SLSQP")
    info = {"success": bool(res.success), "message": res.message, "s_penalty_scale": w[-1]}
    if not res.success:
        return None, info
    return np.asarray(res.x, dtype=float), info


def _solve_weighted_l2_constrained_custom(
    Mmat: np.ndarray,
    b: np.ndarray,
    bounds: List[Tuple[float | None, float | None]],
    weights: np.ndarray,
    *,
    G_ub: np.ndarray | None = None,
    h_ub: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Constrained min 0.5||W u||^2 s.t. M u = b with custom diagonal weights.
    """
    M = np.asarray(Mmat, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = M.shape[1]
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != n:
        raise ValueError("weights length must match number of controls")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")

    def obj(u: np.ndarray) -> float:
        return 0.5 * float((w * u) @ (w * u))

    def jac(u: np.ndarray) -> np.ndarray:
        return (w * w) * u

    cons = [
        {
            "type": "eq",
            "fun": lambda u: M @ u - b,
            "jac": lambda u: M,
        }
    ]
    if G_ub is not None and h_ub is not None and np.asarray(G_ub).size > 0:
        Gv = np.asarray(G_ub, dtype=float)
        hv = np.asarray(h_ub, dtype=float).reshape(-1)
        cons.append(
            {
                "type": "ineq",
                "fun": lambda u, G=Gv, h=hv: h - G @ u,
                "jac": lambda u, G=Gv: -G,
            }
        )

    u0 = np.zeros(n, dtype=float)
    res = minimize(obj, u0, jac=jac, constraints=cons, bounds=bounds, method="SLSQP")
    info = {"success": bool(res.success), "message": res.message}
    if not res.success:
        return None, info
    return np.asarray(res.x, dtype=float), info


def _solve_linf_lp(
    Mmat: np.ndarray,
    b: np.ndarray,
    u_bounds: List[Tuple[float, float]] | None,
    *,
    G_ub: np.ndarray | None = None,
    h_ub: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Solve min t s.t. M u = b and -t <= u_i <= t, s>=0, optional bounds.
    """
    M = np.asarray(Mmat, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = M.shape[1]

    # Decision variable z = [u, t]
    c_obj = np.zeros(n + 1, dtype=float)
    c_obj[-1] = 1.0

    A_eq = np.hstack([M, np.zeros((M.shape[0], 1), dtype=float)])
    b_eq = b

    # Inequalities: u_i - t <= 0 and -u_i - t <= 0
    A_ub = []
    b_ub = []
    for i in range(n):
        row = np.zeros(n + 1, dtype=float)
        row[i] = 1.0
        row[-1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

        row = np.zeros(n + 1, dtype=float)
        row[i] = -1.0
        row[-1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # s >= 0 -> -u_last <= 0
    row = np.zeros(n + 1, dtype=float)
    row[n - 1] = -1.0
    A_ub.append(row)
    b_ub.append(0.0)

    A_ub = np.asarray(A_ub, dtype=float)
    b_ub = np.asarray(b_ub, dtype=float)

    if G_ub is not None and h_ub is not None and np.asarray(G_ub).size > 0:
        Gv = np.asarray(G_ub, dtype=float)
        hv = np.asarray(h_ub, dtype=float).reshape(-1)
        A_ub = np.vstack([A_ub, np.hstack([Gv, np.zeros((Gv.shape[0], 1), dtype=float)])])
        b_ub = np.concatenate([b_ub, hv])

    bounds = _normalize_bounds(u_bounds, n)
    lo, hi = bounds[-1]
    if lo is None or lo < 0.0:
        bounds[-1] = (0.0, hi)
    bounds.append((0.0, None))  # t >= 0

    res = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    info = {"success": bool(res.success), "status": res.status, "message": res.message}
    if not res.success or res.x is None:
        return None, info
    u = np.asarray(res.x[:n], dtype=float)
    return u, info


def _solve_s_coordinate_constrained(
    Mmat: np.ndarray,
    b: np.ndarray,
    u_bounds: List[Tuple[float | None, float | None]],
    *,
    maximize_s: bool,
    G_ub: np.ndarray | None = None,
    h_ub: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    M = np.asarray(Mmat, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = M.shape[1]

    bounds = _normalize_bounds(u_bounds, n)
    lo, hi = bounds[-1]
    if lo is None or lo < 0.0:
        bounds[-1] = (0.0, hi)

    sign = -1.0 if maximize_s else 1.0

    def obj(u: np.ndarray) -> float:
        return float(sign * u[-1])

    def jac(u: np.ndarray) -> np.ndarray:
        grad = np.zeros(n, dtype=float)
        grad[-1] = sign
        return grad

    cons = [
        {
            "type": "eq",
            "fun": lambda u: M @ u - b,
            "jac": lambda u: M,
        }
    ]
    if G_ub is not None and h_ub is not None and np.asarray(G_ub).size > 0:
        Gv = np.asarray(G_ub, dtype=float)
        hv = np.asarray(h_ub, dtype=float).reshape(-1)
        cons.append(
            {
                "type": "ineq",
                "fun": lambda u, G=Gv, h=hv: h - G @ u,
                "jac": lambda u, G=Gv: -G,
            }
        )

    u0 = np.zeros(n, dtype=float)
    res = minimize(obj, u0, jac=jac, constraints=cons, bounds=bounds, method="SLSQP")
    info = {
        "success": bool(res.success),
        "message": res.message,
        "objective": "s_max" if maximize_s else "s_min",
    }
    if not res.success or res.x is None:
        return None, info
    u = np.asarray(res.x, dtype=float)
    info["s_objective_value"] = float(u[-1])
    return u, info


def _normalize_bounds(
    bounds: List[Tuple[float, float]] | None,
    n: int,
) -> List[Tuple[float | None, float | None]]:
    if bounds is None:
        return [(None, None) for _ in range(n)]
    if len(bounds) != n:
        raise ValueError("u_bounds must have length K+1")
    out: List[Tuple[float | None, float | None]] = []
    for lo, hi in bounds:
        out.append((lo, hi))
    return out


def _normalize_bounds_allow_nan(
    bounds: List[Tuple[float, float]] | None,
    n: int,
) -> List[Tuple[float | None, float | None]]:
    if bounds is None:
        return [(None, None) for _ in range(n)]
    if len(bounds) != n:
        raise ValueError("u_bounds must have length K+1")
    out: List[Tuple[float | None, float | None]] = []
    for lo, hi in bounds:
        lo_v = None if lo is None or (isinstance(lo, float) and np.isnan(lo)) else float(lo)
        hi_v = None if hi is None or (isinstance(hi, float) and np.isnan(hi)) else float(hi)
        out.append((lo_v, hi_v))
    return out


def _build_u_bounds_from_blocks(
    dc_bounds: Tuple[float, float] | List[Tuple[float, float]],
    rf_dc_bounds: Tuple[float, float] | List[Tuple[float, float]],
    s_bounds: Tuple[float, float],
    K: int,
    K_rf: int,
) -> List[Tuple[float | None, float | None]]:
    dc_list = _expand_bounds(dc_bounds, K)
    rf_list = _expand_bounds(rf_dc_bounds, K_rf)
    s_lo, s_hi = float(s_bounds[0]), float(s_bounds[1])
    if s_lo < 0.0:
        s_lo = 0.0
    return dc_list + rf_list + [(s_lo, s_hi)]


def _expand_bounds(
    bounds: Tuple[float, float] | List[Tuple[float, float]],
    n: int,
) -> List[Tuple[float | None, float | None]]:
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2 and not isinstance(bounds[0], (list, tuple)):
        lo, hi = bounds
        return [(float(lo), float(hi)) for _ in range(n)]
    if isinstance(bounds, (list, tuple)) and len(bounds) == n:
        return [(float(b[0]), float(b[1])) for b in bounds]
    raise ValueError("bounds must be (low, high) or a list of (low, high) per electrode")


def _solve_avg_max_dc_lp(
    Mmat: np.ndarray,
    b: np.ndarray,
    dc_indices: List[int],
    rf_dc_indices: List[int],
    u_bounds: List[Tuple[float | None, float | None]],
    *,
    G_ub: np.ndarray | None = None,
    h_ub: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    M = np.asarray(Mmat, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = M.shape[1]

    # Variables: [u (n), t_dc, t_rf]
    c_obj = np.zeros(n + 2, dtype=float)
    c_obj[n] = 1.0
    c_obj[n + 1] = 1.0

    A_eq = np.hstack([M, np.zeros((M.shape[0], 2), dtype=float)])
    b_eq = b

    A_ub = []
    b_ub = []
    # dc block
    for i in dc_indices:
        row = np.zeros(n + 2, dtype=float)
        row[i] = 1.0
        row[n] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

        row = np.zeros(n + 2, dtype=float)
        row[i] = -1.0
        row[n] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # rf_dc block
    for j in rf_dc_indices:
        row = np.zeros(n + 2, dtype=float)
        row[j] = 1.0
        row[n + 1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

        row = np.zeros(n + 2, dtype=float)
        row[j] = -1.0
        row[n + 1] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    if len(A_ub) == 0:
        A_ub = None
        b_ub = None
    else:
        A_ub = np.asarray(A_ub, dtype=float)
        b_ub = np.asarray(b_ub, dtype=float)

    if G_ub is not None and h_ub is not None and np.asarray(G_ub).size > 0:
        Gv = np.asarray(G_ub, dtype=float)
        hv = np.asarray(h_ub, dtype=float).reshape(-1)
        G_aug = np.hstack([Gv, np.zeros((Gv.shape[0], 2), dtype=float)])
        if A_ub is None:
            A_ub = G_aug
            b_ub = hv
        else:
            A_ub = np.vstack([A_ub, G_aug])
            b_ub = np.concatenate([b_ub, hv])

    bounds = _normalize_bounds(u_bounds, n)
    lo, hi = bounds[-1]
    if lo is None or lo < 0.0:
        bounds[-1] = (0.0, hi)
    bounds.append((0.0, None))  # t_dc
    bounds.append((0.0, None))  # t_rf

    res = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    info = {
        "success": bool(res.success),
        "status": res.status,
        "message": res.message,
    }
    if not res.success or res.x is None:
        return None, info
    t_dc = float(res.x[n])
    t_rf = float(res.x[n + 1])
    info["u_norminf_dc"] = t_dc
    info["u_norminf_rf_dc"] = t_rf
    info["objective_value"] = 0.5 * (t_dc + t_rf)
    u = np.asarray(res.x[:n], dtype=float)
    return u, info
