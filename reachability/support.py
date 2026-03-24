"""
Support-query solve for modal-curvature reachability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import ReachabilityModel

try:
    from scipy.optimize import linprog
except Exception:  # pragma: no cover - optional dependency
    linprog = None


@dataclass
class SupportQueryResult:
    """
    Result of one support query in modal-curvature space.
    """

    success: bool
    status: str
    message: str
    solver_status: int | None
    direction: np.ndarray
    direction_unit: np.ndarray
    objective_value: float
    objective_value_unit_direction: float
    u_star: np.ndarray | None
    lambda_star: np.ndarray | None


def solve_reachability_support_query(
    model: ReachabilityModel,
    direction: Sequence[float],
    *,
    method: str = "highs",
) -> SupportQueryResult:
    """
    Solve one support query for the reachable modal-curvature set.

    For direction d in R^3, this solves:
        maximize    d^T (T u)
        subject to  E u = e
                    lower_u <= u <= upper_u

    using an LP in control variables u.
    """
    if linprog is None:
        raise RuntimeError("SciPy is required for support-query LP solves")

    d = np.asarray(direction, dtype=float).reshape(-1)
    if d.shape[0] != 3:
        raise ValueError("direction must have length 3")
    if not np.all(np.isfinite(d)):
        raise ValueError("direction must be finite")

    d_norm = float(np.linalg.norm(d))
    if d_norm == 0.0:
        raise ValueError("direction must be nonzero")
    d_unit = d / d_norm

    # LP minimization form: minimize -d^T(Tu) == minimize -(T^T d)^T u.
    c_obj = -(model.T.T @ d_unit)
    res = linprog(
        c=c_obj,
        A_eq=model.E,
        b_eq=model.e,
        bounds=model.bounds_tuples(),
        method=method,
    )

    if not bool(res.success) or res.x is None:
        return SupportQueryResult(
            success=False,
            status=_status_from_solver_code(int(res.status)),
            message=str(res.message),
            solver_status=int(res.status),
            direction=d,
            direction_unit=d_unit,
            objective_value=float("nan"),
            objective_value_unit_direction=float("nan"),
            u_star=None,
            lambda_star=None,
        )

    u_star = np.asarray(res.x, dtype=float)
    lambda_star = model.T @ u_star
    objective_value = float(d @ lambda_star)
    objective_value_unit = float(d_unit @ lambda_star)
    return SupportQueryResult(
        success=True,
        status="ok",
        message=str(res.message),
        solver_status=int(res.status),
        direction=d,
        direction_unit=d_unit,
        objective_value=objective_value,
        objective_value_unit_direction=objective_value_unit,
        u_star=u_star,
        lambda_star=lambda_star,
    )


def _status_from_solver_code(code: int) -> str:
    if code == 2:
        return "infeasible"
    if code == 3:
        return "unbounded"
    return "solver_failed"

