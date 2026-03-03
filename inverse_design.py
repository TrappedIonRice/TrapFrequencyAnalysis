"""
Inverse design entrypoint: map (r0, freqs, principal directions) to controls u.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import constants

from control_constraints import build_L_b_for_point, build_target_hessian
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
    freqs: np.ndarray,
    principal_dirs: np.ndarray,
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
) -> Dict[str, object]:
    """
    Solve for control vector u that enforces target equilibrium and Hessian.
    """
    r0v = np.asarray(r0, dtype=float).reshape(-1)
    if r0v.shape[0] != 3:
        raise ValueError("r0 must have shape (3,)")
    freqs_v = np.asarray(freqs, dtype=float).reshape(-1)
    if freqs_v.shape[0] != 3:
        raise ValueError("freqs must have shape (3,)")

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

    # Build target Hessian and constraints
    Kstar = build_target_hessian(
        freqs_v,
        principal_dirs,
        mass=ion_mass_kg,
        charge=ion_charge_c,
        poly_is_potential_energy=poly_is_potential_energy,
        freqs_in_hz=freqs_in_hz,
    )
    L, b = build_L_b_for_point(powers, r0v, Kstar, include_gradient=True)
    Mmat = L @ A

    # Solve for u
    solver_info: Dict[str, object] = {}
    status = "ok"
    if objective not in ("l2", "linf", "weighted_l2", "avg_max_dc", "l2_dc"):
        raise ValueError("objective must be 'l2', 'linf', 'weighted_l2', 'avg_max_dc', or 'l2_dc'")

    bounds_requested = enforce_bounds or (u_bounds is not None)
    if objective == "l2":
        if bounds_requested:
            if not _HAVE_SCIPY:
                raise RuntimeError("SciPy is required for constrained l2 solve")
            if u_bounds is None:
                u_bounds = _build_u_bounds_from_blocks(dc_bounds, rf_dc_bounds, s_bounds, len(dc_electrodes), len(rf_dc_electrodes))
            u, info = _solve_l2_constrained(Mmat, b, u_bounds)
            solver_info.update(info)
            if u is None:
                status = "qp_failed"
        else:
            u, info = _solve_l2_min_norm(Mmat, b, ridge_lambda=ridge_lambda)
            solver_info.update(info)
            if info.get("used_ridge", False):
                status = "used_ridge"
    elif objective == "weighted_l2":
        if bounds_requested:
            if not _HAVE_SCIPY:
                raise RuntimeError("SciPy is required for constrained weighted_l2 solve")
            if u_bounds is None:
                u_bounds = _build_u_bounds_from_blocks(dc_bounds, rf_dc_bounds, s_bounds, len(dc_electrodes), len(rf_dc_electrodes))
            u, info = _solve_weighted_l2_constrained(Mmat, b, u_bounds, s_penalty_scale)
            solver_info.update(info)
            if u is None:
                status = "qp_failed"
        else:
            u, info = _solve_weighted_l2_min_norm(
                Mmat, b, s_penalty_scale=s_penalty_scale, ridge_lambda=ridge_lambda
            )
            solver_info.update(info)
            if info.get("used_ridge", False):
                status = "used_ridge"
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
        u, info = _solve_weighted_l2_constrained_custom(Mmat, b, bounds_clean, weights)
        solver_info.update(info)
        if u is None:
            status = "qp_failed"
    elif objective == "avg_max_dc":
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy is required for avg_max_dc solve")
        K = len(dc_electrodes)
        K_rf = len(rf_dc_electrodes)
        dc_indices = list(range(0, K))
        rf_dc_indices = list(range(K, K + K_rf))
        if u_bounds is None:
            u_bounds = _build_u_bounds_from_blocks(dc_bounds, rf_dc_bounds, s_bounds, K, K_rf)
        u, info = _solve_avg_max_dc_lp(Mmat, b, dc_indices, rf_dc_indices, u_bounds)
        solver_info.update(info)
        if u is None:
            status = "lp_failed"
    else:
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy is required for linf solve")
        u, info = _solve_linf_lp(Mmat, b, u_bounds)
        solver_info.update(info)
        if u is None:
            status = "lp_failed"

    # Diagnostics
    if u is None:
        print(f"[solve_u_for_targets] Bounded solve failed: {solver_info.get('message', '')}")
        return {
            "u": None,
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
            "cache_hit": out.get("cache_hit", False),
            "cache_path": out.get("cache_path"),
            "cfg": out.get("cfg"),
        }

    if np.all(np.isfinite(u)):
        resid = Mmat @ u - b
        residual_norm = float(np.linalg.norm(resid))
        eq_tol = 1e-9 * (1.0 + float(np.linalg.norm(b)))
        solver_info["eq_tol"] = eq_tol
        if residual_norm > eq_tol:
            solver_info["eq_resid"] = residual_norm
            if bounds_requested:
                status = "eq_violation"
                print(f"[solve_u_for_targets] Equality residual {residual_norm:.3e} exceeds tol {eq_tol:.3e}")
                return {
                    "u": None,
                    "u_dc": None,
                    "u_rf_dc": None,
                    "u_s": None,
                    "A": A,
                    "powers": powers,
                    "L": L,
                    "b": b,
                    "M": Mmat,
                    "residual_norm": residual_norm,
                    "u_norm2": None,
                    "u_norminf": None,
                    "status": status,
                    "solver_info": solver_info,
                    "cache_hit": out.get("cache_hit", False),
                    "cache_path": out.get("cache_path"),
                    "cfg": out.get("cfg"),
                }
            status = "best_effort_large_resid"
        u_norm2 = float(np.linalg.norm(u))
        u_norminf = float(np.max(np.abs(u)))
        s_val = float(u[-1])
        if s_val < 0 and status == "ok":
            status = "best_effort_s_negative"
    else:
        # Fall back to unconstrained least-squares residual for feasibility signal.
        u_lsq, *_ = np.linalg.lstsq(Mmat, b, rcond=None)
        resid = Mmat @ u_lsq - b
        residual_norm = float(np.linalg.norm(resid))
        eq_tol = 1e-9 * (1.0 + float(np.linalg.norm(b)))
        solver_info["eq_tol"] = eq_tol
        u_norm2 = float("nan")
        u_norminf = float("nan")
        s_val = float("nan")
        solver_info["lsq_residual_norm"] = residual_norm

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
        "residual_norm": residual_norm,
        "u_norm2": u_norm2,
        "u_norminf": u_norminf,
        "status": status,
        "solver_info": solver_info,
        "cache_hit": out.get("cache_hit", False),
        "cache_path": out.get("cache_path"),
        "cfg": out.get("cfg"),
    }


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

    cons = {
        "type": "eq",
        "fun": lambda u: M @ u - b,
        "jac": lambda u: M,
    }

    u0 = np.zeros(n, dtype=float)
    res = minimize(obj, u0, jac=jac, constraints=[cons], bounds=bounds, method="SLSQP")
    info = {"success": bool(res.success), "message": res.message}
    if not res.success:
        return None, info
    return np.asarray(res.x, dtype=float), info


def _solve_weighted_l2_constrained(
    Mmat: np.ndarray,
    b: np.ndarray,
    u_bounds: List[Tuple[float, float]] | None,
    s_penalty_scale: float,
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

    cons = {
        "type": "eq",
        "fun": lambda u: M @ u - b,
        "jac": lambda u: M,
    }

    u0 = np.zeros(n, dtype=float)
    res = minimize(obj, u0, jac=jac, constraints=[cons], bounds=bounds, method="SLSQP")
    info = {"success": bool(res.success), "message": res.message, "s_penalty_scale": w[-1]}
    if not res.success:
        return None, info
    return np.asarray(res.x, dtype=float), info


def _solve_weighted_l2_constrained_custom(
    Mmat: np.ndarray,
    b: np.ndarray,
    bounds: List[Tuple[float | None, float | None]],
    weights: np.ndarray,
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

    cons = {
        "type": "eq",
        "fun": lambda u: M @ u - b,
        "jac": lambda u: M,
    }

    u0 = np.zeros(n, dtype=float)
    res = minimize(obj, u0, jac=jac, constraints=[cons], bounds=bounds, method="SLSQP")
    info = {"success": bool(res.success), "message": res.message}
    if not res.success:
        return None, info
    return np.asarray(res.x, dtype=float), info


def _solve_linf_lp(
    Mmat: np.ndarray,
    b: np.ndarray,
    u_bounds: List[Tuple[float, float]] | None,
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
