from __future__ import annotations

"""Standalone unrestricted harmonic 2D minimizer for the playground.

This module refines full unconstrained x-z coordinates under the same harmonic
trap model used by `SymmetryEnforced_Harmonic2D`:

    U(x, z) = sum_i (a_x x_i^2 + a_z z_i^2) + sum_{i<j} 1 / r_ij

with no symmetry constraints on the ions.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import basinhopping, minimize

from equilibrium_playground.reduced_model import pack_xz, split_xz
from equilibrium_playground.symmetry_enforced_harmonic2d import (
    HarmonicTrap2D,
    SymmetryEnforcedHarmonic2DResult,
    normalize_harmonic_trap,
    solve_symmetry_enforced_harmonic2d,
)


_EPS = 1.0e-24


@dataclass
class UnrestrictedHarmonic2DResult:
    """Result bundle for the standalone unrestricted harmonic 2D solver."""

    positions_2d: np.ndarray
    q: np.ndarray
    energy: float
    success: bool
    message: str
    optimizer_path: str
    trap_input_mode: str
    trap_parameters: dict[str, float | str]
    total_ion_count: int
    seed_positions_2d: np.ndarray
    seed_q: np.ndarray
    seed_to_final_rms_move: float
    gradient_max_norm: float
    use_basinhopping: bool
    nfev: int | None = None
    njev: int | None = None
    nit: int | None = None
    basin_iterations: int | None = None
    basinhopping_success: bool | None = None
    basinhopping_message: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "minimizer_name": "Unrestricted_Harmonic2D",
            "optimizer_name": self.optimizer_path,
            "optimizer_path": self.optimizer_path,
            "energy": float(self.energy),
            "success": bool(self.success),
            "message": str(self.message),
            "total_ion_count": int(self.total_ion_count),
            "gradient_max_norm": float(self.gradient_max_norm),
            "distance_moved_from_seed_dimless": float(self.seed_to_final_rms_move),
            "trap_input_mode": self.trap_input_mode,
            "trap_parameters": dict(self.trap_parameters),
            "positions_2d": self.positions_2d.copy(),
        }


@dataclass(frozen=True)
class SymmetrySeedToUnrestrictedComparisonResult:
    """Comparison between the symmetry solution and unrestricted refinement."""

    symmetry_result: SymmetryEnforcedHarmonic2DResult
    unrestricted_result: UnrestrictedHarmonic2DResult
    symmetry_energy: float
    unrestricted_energy: float
    delta_energy: float
    rms_displacement_sorted: float
    unrestricted_gradient_max_norm: float
    unrestricted_success: bool
    unrestricted_message: str
    total_ion_count: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "comparison_name": "SymmetrySeedToUnrestrictedRefinement",
            "symmetry_energy": float(self.symmetry_energy),
            "unrestricted_energy": float(self.unrestricted_energy),
            "delta_energy": float(self.delta_energy),
            "rms_displacement_sorted": float(self.rms_displacement_sorted),
            "unrestricted_gradient_max_norm": float(self.unrestricted_gradient_max_norm),
            "unrestricted_success": bool(self.unrestricted_success),
            "unrestricted_message": str(self.unrestricted_message),
            "total_ion_count": int(self.total_ion_count),
        }


def _positions_to_points_2d(seed_positions_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(seed_positions_2d, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError("seed_positions_2d must have shape (N, 2)")
        return arr.copy()

    flat = arr.reshape(-1)
    if flat.size % 2 != 0:
        raise ValueError("flat seed vector must contain an even number of entries")
    x, z = split_xz(flat)
    return np.column_stack((x, z))


def _points_2d_to_q(points_2d: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    return pack_xz(pts[:, 0], pts[:, 1])


def _q_to_points_2d(q: np.ndarray) -> np.ndarray:
    x, z = split_xz(q)
    return np.column_stack((x, z))


def _safe_radius(dx: float, dz: float) -> float:
    return float(np.sqrt(max(dx * dx + dz * dz, _EPS)))


def _gradient_max_norm(gradient_q: np.ndarray) -> float:
    grad = np.asarray(gradient_q, dtype=float).reshape(-1)
    if grad.size == 0:
        return 0.0
    return float(np.max(np.abs(grad)))


def _sort_positions_xz(points_2d: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[order]


def sorted_rms_displacement(points_a_2d: np.ndarray, points_b_2d: np.ndarray) -> float:
    """Return the RMS displacement after lexicographic `(x, z)` sorting."""

    pts_a = _sort_positions_xz(points_a_2d)
    pts_b = _sort_positions_xz(points_b_2d)
    if pts_a.shape != pts_b.shape:
        raise ValueError("point clouds must have the same shape")
    diff = pts_a - pts_b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def unrestricted_trap_energy(q: np.ndarray, trap: HarmonicTrap2D) -> float:
    """Return the harmonic trap energy for unrestricted x-z coordinates."""

    x, z = split_xz(q)
    return float(np.sum(trap.a_x * x * x + trap.a_z * z * z))


def unrestricted_trap_gradient(q: np.ndarray, trap: HarmonicTrap2D) -> np.ndarray:
    """Return the analytic harmonic trap gradient."""

    x, z = split_xz(q)
    return pack_xz(2.0 * trap.a_x * x, 2.0 * trap.a_z * z)


def unrestricted_coulomb_energy(q: np.ndarray) -> float:
    """Return the Coulomb repulsion energy for unrestricted x-z coordinates."""

    x, z = split_xz(q)
    n_ions = x.size
    energy = 0.0
    for i in range(n_ions):
        for j in range(i + 1, n_ions):
            energy += 1.0 / _safe_radius(float(x[i] - x[j]), float(z[i] - z[j]))
    return float(energy)


def unrestricted_coulomb_gradient(q: np.ndarray) -> np.ndarray:
    """Return the analytic Coulomb gradient for unrestricted x-z coordinates."""

    x, z = split_xz(q)
    n_ions = x.size
    grad_x = np.zeros(n_ions, dtype=float)
    grad_z = np.zeros(n_ions, dtype=float)
    for i in range(n_ions):
        for j in range(i + 1, n_ions):
            dx = float(x[i] - x[j])
            dz = float(z[i] - z[j])
            r2 = max(dx * dx + dz * dz, _EPS)
            r3 = r2 * np.sqrt(r2)
            contrib_x = dx / r3
            contrib_z = dz / r3
            grad_x[i] -= contrib_x
            grad_z[i] -= contrib_z
            grad_x[j] += contrib_x
            grad_z[j] += contrib_z
    return pack_xz(grad_x, grad_z)


def unrestricted_total_energy(q: np.ndarray, trap: HarmonicTrap2D) -> float:
    """Return the full unrestricted harmonic 2D energy."""

    return unrestricted_trap_energy(q, trap) + unrestricted_coulomb_energy(q)


def unrestricted_total_gradient(q: np.ndarray, trap: HarmonicTrap2D) -> np.ndarray:
    """Return the full analytic unrestricted harmonic 2D gradient."""

    return unrestricted_trap_gradient(q, trap) + unrestricted_coulomb_gradient(q)


def solve_unrestricted_harmonic2d(
    seed_positions_2d: np.ndarray,
    *,
    omega_x: float | None = None,
    omega_z: float | None = None,
    alpha: float | None = None,
    scale: float | None = None,
    use_basinhopping: bool = False,
    basinhopping_niter: int = 30,
    basin_temperature: float = 0.35,
    basin_stepsize: float = 0.3,
    bfgs_gtol: float = 1.0e-10,
    bfgs_maxiter: int = 1000,
    local_gtol: float = 1.0e-11,
    local_ftol: float = 1.0e-14,
    local_maxiter: int = 4000,
) -> UnrestrictedHarmonic2DResult:
    """Run the standalone unrestricted harmonic 2D minimizer.

    The default mode is local-only refinement from the supplied full-coordinate
    seed. If `use_basinhopping=True`, a basin-hopping stage is run before the
    final local refinement.
    """

    seed_points = _positions_to_points_2d(seed_positions_2d)
    seed_q = _points_2d_to_q(seed_points)
    trap = normalize_harmonic_trap(
        omega_x=omega_x,
        omega_z=omega_z,
        alpha=alpha,
        scale=scale,
    )

    energy_fn = lambda q: unrestricted_total_energy(q, trap)
    gradient_fn = lambda q: unrestricted_total_gradient(q, trap)

    current_q = seed_q.copy()
    optimizer_path = "L-BFGS-B"
    basin_iterations = None
    basinhopping_success = None
    basinhopping_message = None

    if bool(use_basinhopping):
        optimizer_path = "basinhopping(L-BFGS-B)->L-BFGS-B"
        bh_kwargs = {
            "method": "L-BFGS-B",
            "jac": gradient_fn,
            "options": {
                "gtol": float(bfgs_gtol),
                "maxiter": int(bfgs_maxiter),
            },
        }
        bh_res = basinhopping(
            energy_fn,
            current_q,
            minimizer_kwargs=bh_kwargs,
            niter=int(basinhopping_niter),
            T=float(basin_temperature),
            stepsize=float(basin_stepsize),
            disp=False,
        )
        current_q = np.asarray(bh_res.x, dtype=float)
        basin_iterations = int(basinhopping_niter)
        bh_lowest = getattr(bh_res, "lowest_optimization_result", None)
        basinhopping_success = (
            bool(getattr(bh_lowest, "success", False)) if bh_lowest is not None else None
        )
        basinhopping_message = str(getattr(bh_lowest, "message", getattr(bh_res, "message", "")))

    refine_res = minimize(
        energy_fn,
        current_q,
        jac=gradient_fn,
        method="L-BFGS-B",
        options={
            "gtol": float(local_gtol),
            "ftol": float(local_ftol),
            "maxiter": int(local_maxiter),
            "maxls": 50,
        },
    )

    final_q = (
        np.asarray(refine_res.x, dtype=float)
        if getattr(refine_res, "x", None) is not None
        else np.asarray(current_q, dtype=float)
    )
    final_points = _q_to_points_2d(final_q)
    final_gradient = gradient_fn(final_q)

    return UnrestrictedHarmonic2DResult(
        positions_2d=final_points,
        q=final_q.copy(),
        energy=float(getattr(refine_res, "fun", energy_fn(final_q))),
        success=bool(getattr(refine_res, "success", False)),
        message=str(getattr(refine_res, "message", "")),
        optimizer_path=optimizer_path,
        trap_input_mode=trap.input_mode,
        trap_parameters=trap.to_metadata(),
        total_ion_count=int(final_points.shape[0]),
        seed_positions_2d=seed_points.copy(),
        seed_q=seed_q.copy(),
        seed_to_final_rms_move=sorted_rms_displacement(seed_points, final_points),
        gradient_max_norm=_gradient_max_norm(final_gradient),
        use_basinhopping=bool(use_basinhopping),
        nfev=getattr(refine_res, "nfev", None),
        njev=getattr(refine_res, "njev", None),
        nit=getattr(refine_res, "nit", None),
        basin_iterations=basin_iterations,
        basinhopping_success=basinhopping_success,
        basinhopping_message=basinhopping_message,
    )


def compare_symmetry_seed_to_unrestricted_refinement(
    num_ions_vert: int,
    num_ions_hor: int,
    num_ions_free: int,
    *,
    omega_x: float | None = None,
    omega_z: float | None = None,
    alpha: float | None = None,
    scale: float | None = None,
    symmetry_solver_kwargs: dict[str, Any] | None = None,
    unrestricted_solver_kwargs: dict[str, Any] | None = None,
) -> SymmetrySeedToUnrestrictedComparisonResult:
    """Compare the symmetry solution to unrestricted refinement from that seed."""

    symmetry_kwargs = {} if symmetry_solver_kwargs is None else dict(symmetry_solver_kwargs)
    unrestricted_kwargs = (
        {} if unrestricted_solver_kwargs is None else dict(unrestricted_solver_kwargs)
    )

    symmetry_result = solve_symmetry_enforced_harmonic2d(
        num_ions_vert,
        num_ions_hor,
        num_ions_free,
        omega_x=omega_x,
        omega_z=omega_z,
        alpha=alpha,
        scale=scale,
        **symmetry_kwargs,
    )

    unrestricted_result = solve_unrestricted_harmonic2d(
        symmetry_result.positions_2d,
        omega_x=omega_x,
        omega_z=omega_z,
        alpha=alpha,
        scale=scale,
        **unrestricted_kwargs,
    )

    return SymmetrySeedToUnrestrictedComparisonResult(
        symmetry_result=symmetry_result,
        unrestricted_result=unrestricted_result,
        symmetry_energy=float(symmetry_result.energy),
        unrestricted_energy=float(unrestricted_result.energy),
        delta_energy=float(unrestricted_result.energy - symmetry_result.energy),
        rms_displacement_sorted=sorted_rms_displacement(
            symmetry_result.positions_2d,
            unrestricted_result.positions_2d,
        ),
        unrestricted_gradient_max_norm=float(unrestricted_result.gradient_max_norm),
        unrestricted_success=bool(unrestricted_result.success),
        unrestricted_message=str(unrestricted_result.message),
        total_ion_count=int(unrestricted_result.total_ion_count),
    )
