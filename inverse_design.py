"""
Inverse design entrypoint: map (r0, freqs, principal directions) to controls u.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from control_constraints import build_L_b_for_point, build_target_hessian
from linearsyssolve101 import build_voltage_to_c_matrix

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
    rf_freq_hz: float,
    num_samples: int,
    dc_bounds: Tuple[float, float] | List[Tuple[float, float]] = (-500.0, 500.0),
    rf_dc_bounds: Tuple[float, float] | List[Tuple[float, float]] = (-50.0, 50.0),
    rf2_bounds: Tuple[float, float] = (0.0, (5000.0**2)),
    polyfit_deg: int = 4,
    seed: int = 0,
    objective: str = "l2",
    enforce_bounds: bool = False,
    u_bounds: List[Tuple[float, float]] | None = None,
    ridge_lambda: float = 0.0,
    rf2_penalty_scale: float = 1e-5,
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
    out = build_voltage_to_c_matrix(
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
    if objective not in ("l2", "linf", "weighted_l2"):
        raise ValueError("objective must be 'l2', 'linf', or 'weighted_l2'")

    if objective == "l2":
        if enforce_bounds or _bounds_needed_for_rf2(u_bounds):
            if not _HAVE_SCIPY:
                raise RuntimeError("SciPy is required for constrained l2 solve")
            u, info = _solve_l2_constrained(Mmat, b, u_bounds)
            solver_info.update(info)
        else:
            u, info = _solve_l2_min_norm(Mmat, b, ridge_lambda=ridge_lambda)
            solver_info.update(info)
            if info.get("used_ridge", False):
                status = "used_ridge"
    elif objective == "weighted_l2":
        if enforce_bounds or _bounds_needed_for_rf2(u_bounds):
            if not _HAVE_SCIPY:
                raise RuntimeError("SciPy is required for constrained weighted_l2 solve")
            u, info = _solve_weighted_l2_constrained(Mmat, b, u_bounds, rf2_penalty_scale)
            solver_info.update(info)
        else:
            u, info = _solve_weighted_l2_min_norm(
                Mmat, b, rf2_penalty_scale=rf2_penalty_scale, ridge_lambda=ridge_lambda
            )
            solver_info.update(info)
            if info.get("used_ridge", False):
                status = "used_ridge"
    else:
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy is required for linf solve")
        u, info = _solve_linf_lp(Mmat, b, u_bounds)
        solver_info.update(info)
        if not info.get("success", True):
            status = "lp_failed"

    # Diagnostics
    resid = Mmat @ u - b
    residual_norm = float(np.linalg.norm(resid))
    u_norm2 = float(np.linalg.norm(u))
    u_norminf = float(np.max(np.abs(u)))
    rf2 = float(u[-1])
    rf_amp = float(np.sqrt(max(rf2, 0.0)))
    if rf2 < 0 and status == "ok":
        status = "rf2_negative"

    return {
        "u": u,
        "u_dc": u[: len(dc_electrodes)],
        "u_rf_dc": u[len(dc_electrodes) : len(dc_electrodes) + len(rf_dc_electrodes)],
        "u_rf2": rf2,
        "u_rf_amp": rf_amp,
        "A": A,
        "powers": powers,
        "L": L,
        "b": b,
        "M": Mmat,
        "residual_norm": residual_norm,
        "u_norm2": u_norm2,
        "u_norminf": u_norminf,
        "rf2": rf2,
        "rf_amp": rf_amp,
        "status": status,
        "solver_info": solver_info,
    }


def _bounds_needed_for_rf2(u_bounds: List[Tuple[float, float]] | None) -> bool:
    if u_bounds is None:
        return True
    if len(u_bounds) == 0:
        return True
    lo, _ = u_bounds[-1]
    return lo is None or lo < 0.0


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
    rf2_penalty_scale: float,
    ridge_lambda: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Solve min ||W u||_2 s.t. M u = b, where W is diagonal and rf2 has lower weight.
    rf2_penalty_scale < 1 means rf2 is penalized less than DC entries.
    """
    info: Dict[str, object] = {}
    M = np.asarray(Mmat, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = M.shape[1]

    w = np.ones(n, dtype=float)
    w[-1] = float(rf2_penalty_scale)
    if w[-1] <= 0.0:
        raise ValueError("rf2_penalty_scale must be positive")

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
        info["rf2_penalty_scale"] = w[-1]
        return u, info

    MMt = Mtilde @ Mtilde.T
    try:
        x = np.linalg.solve(MMt, b)
        v = Mtilde.T @ x
        u = Winv @ v
        info["used_ridge"] = False
        info["rf2_penalty_scale"] = w[-1]
        return u, info
    except np.linalg.LinAlgError:
        lam = ridge_lambda if ridge_lambda > 0 else 1e-8
        x = np.linalg.solve(MMt + lam * np.eye(MMt.shape[0]), b)
        v = Mtilde.T @ x
        u = Winv @ v
        info["used_ridge"] = True
        info["ridge_lambda"] = lam
        info["rf2_penalty_scale"] = w[-1]
        return u, info


def _solve_l2_constrained(
    Mmat: np.ndarray,
    b: np.ndarray,
    u_bounds: List[Tuple[float, float]] | None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Constrained min 0.5||u||^2 s.t. M u = b and bounds (including rf2>=0).
    """
    M = np.asarray(Mmat, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = M.shape[1]

    bounds = _normalize_bounds(u_bounds, n)
    # Enforce rf2 >= 0
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
    return np.asarray(res.x, dtype=float), info


def _solve_weighted_l2_constrained(
    Mmat: np.ndarray,
    b: np.ndarray,
    u_bounds: List[Tuple[float, float]] | None,
    rf2_penalty_scale: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Constrained min 0.5||W u||^2 s.t. M u = b and bounds (including rf2>=0).
    """
    M = np.asarray(Mmat, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = M.shape[1]

    w = np.ones(n, dtype=float)
    w[-1] = float(rf2_penalty_scale)
    if w[-1] <= 0.0:
        raise ValueError("rf2_penalty_scale must be positive")

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
    info = {"success": bool(res.success), "message": res.message, "rf2_penalty_scale": w[-1]}
    return np.asarray(res.x, dtype=float), info


def _solve_linf_lp(
    Mmat: np.ndarray,
    b: np.ndarray,
    u_bounds: List[Tuple[float, float]] | None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Solve min t s.t. M u = b and -t <= u_i <= t, rf2>=0, optional bounds.
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

    # rf2 >= 0 -> -u_last <= 0
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
