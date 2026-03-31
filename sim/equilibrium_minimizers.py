from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import BFGS, Bounds, basinhopping, minimize

import constants
from sim.equilibrium_seeders import (
    PlaneDefinition,
    build_cryo2dcopy_seed_context,
    build_quartic2d_seed_context,
    generate_quartic2d_seed_candidates,
    get_default_quartic2d_plane,
    get_quartic2d_planar_bounds_arrays,
    make_cryo2dcopy_triangular_seed,
    sort_positions_on_plane,
)


QUARTIC2D_101 = "Quartic2D_101"
CRYO2DCOPY = "cryo2dcopy"
_PROJECTED_GRADIENT_SUCCESS_TOL = 1.0e-6


def _minimum_pair_separation(positions: np.ndarray) -> float | None:
    pts = np.asarray(positions, dtype=float).reshape(-1, 3)
    if len(pts) < 2:
        return None
    best = np.inf
    for idx in range(len(pts)):
        diffs = pts[idx + 1 :] - pts[idx]
        if diffs.size == 0:
            continue
        dists = np.linalg.norm(diffs, axis=1)
        best = min(best, float(np.min(dists)))
    return None if not np.isfinite(best) else float(best)


def _attempt_sort_key(attempt: "_OptimizationAttempt") -> float:
    return attempt.energy if np.isfinite(attempt.energy) else np.inf


def _best_attempt(attempts: list["_OptimizationAttempt"]) -> "_OptimizationAttempt":
    successful = [
        attempt
        for attempt in attempts
        if attempt.success and np.isfinite(attempt.energy)
    ]
    if successful:
        return min(successful, key=_attempt_sort_key)
    return min(
        attempts,
        key=lambda attempt: attempt.energy if np.isfinite(attempt.energy) else np.inf,
    )


def _max_projected_gradient_norm(
    objective: "_PlanarEquilibriumObjective",
    flat_coords: np.ndarray,
) -> float:
    gradient = np.asarray(objective.gradient(np.asarray(flat_coords, dtype=float)), dtype=float)
    if gradient.size == 0:
        return 0.0
    return float(np.max(np.abs(gradient)))


def _distance_moved_from_seed_dimless(
    final_positions_2d: np.ndarray,
    seed_positions_2d: np.ndarray | None,
) -> float | None:
    if seed_positions_2d is None:
        return None
    final_pts = np.asarray(final_positions_2d, dtype=float).reshape(-1, 2)
    seed_pts = np.asarray(seed_positions_2d, dtype=float).reshape(-1, 2)
    if final_pts.shape != seed_pts.shape or final_pts.size == 0:
        return None
    displacement = final_pts - seed_pts
    return float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))))


def _lbfgsb_result_sort_key(result: Any, objective: "_PlanarEquilibriumObjective") -> tuple[float, float]:
    x_val = getattr(result, "x", None)
    if x_val is None:
        return (np.inf, np.inf)
    fun_val = float(getattr(result, "fun", np.inf))
    grad_norm = _max_projected_gradient_norm(objective, np.asarray(x_val, dtype=float))
    return (
        fun_val if np.isfinite(fun_val) else np.inf,
        grad_norm,
    )


def _run_lbfgsb_with_restarts(
    objective: "_PlanarEquilibriumObjective",
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    *,
    options: dict[str, Any],
    max_restarts: int,
) -> tuple[Any, int, int, int]:
    x_start = np.asarray(x0, dtype=float).reshape(-1)
    best_result = None
    total_nfev = 0
    total_njev = 0
    total_nit = 0

    for _ in range(int(max_restarts) + 1):
        result = minimize(
            objective.energy,
            x_start,
            method="L-BFGS-B",
            jac=objective.gradient,
            bounds=bounds,
            options=options,
        )
        total_nfev += int(getattr(result, "nfev", 0) or 0)
        total_njev += int(getattr(result, "njev", 0) or 0)
        total_nit += int(getattr(result, "nit", 0) or 0)

        if best_result is None or _lbfgsb_result_sort_key(result, objective) < _lbfgsb_result_sort_key(
            best_result,
            objective,
        ):
            best_result = result

        x_val = getattr(result, "x", None)
        if x_val is None:
            break
        grad_norm = _max_projected_gradient_norm(objective, np.asarray(x_val, dtype=float))
        if grad_norm <= _PROJECTED_GRADIENT_SUCCESS_TOL:
            break
        x_start = np.asarray(x_val, dtype=float)

    return best_result, total_nfev, total_njev, total_nit


@dataclass
class _OptimizationAttempt:
    positions_2d: np.ndarray
    positions_3d_dimless: np.ndarray
    energy: float
    success: bool
    message: str
    optimizer_name: str
    stage: str
    seed_family: str | None
    seed_positions_2d: np.ndarray | None
    seed_positions_3d_dimless: np.ndarray | None
    seed_spacing: float | None = None
    seed_fits_bounds_directly: bool | None = None
    projected_gradient_max_norm: float | None = None
    distance_moved_from_seed_dimless: float | None = None
    nfev: int | None = None
    njev: int | None = None
    nit: int | None = None
    basin_iterations: int | None = None


@dataclass
class EquilibriumSearchResult:
    positions: np.ndarray
    positions_dimless: np.ndarray
    energy: float
    success: bool
    message: str
    optimizer_name: str
    minimizer_name: str
    plane_normal: np.ndarray
    seed_family: str | None = None
    seed_positions: np.ndarray | None = None
    seed_positions_dimless: np.ndarray | None = None
    seed_spacing: float | None = None
    seed_fits_bounds_directly: bool | None = None
    seed_jitter: float | None = None
    seed_rng_seed: int | None = None
    alpha_eff: float | None = None
    reference_point_dimless: np.ndarray | None = None
    stage: str = "local"
    optimizer_path: str | None = None
    projected_gradient_max_norm: float | None = None
    distance_moved_from_seed_dimless: float | None = None
    nfev: int | None = None
    njev: int | None = None
    nit: int | None = None
    basin_iterations: int | None = None
    num_local_attempts: int = 0
    num_basin_attempts: int = 0
    local_top_energies: np.ndarray | None = None
    minimum_pair_separation: float | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "positions": np.asarray(self.positions, dtype=float),
            "positions_dimless": np.asarray(self.positions_dimless, dtype=float),
            "energy": float(self.energy),
            "success": bool(self.success),
            "message": str(self.message),
            "optimizer_name": str(self.optimizer_name),
            "minimizer_name": str(self.minimizer_name),
            "plane_normal": np.asarray(self.plane_normal, dtype=float),
            "seed_family": self.seed_family,
            "seed_positions": (
                None
                if self.seed_positions is None
                else np.asarray(self.seed_positions, dtype=float)
            ),
            "seed_positions_dimless": (
                None
                if self.seed_positions_dimless is None
                else np.asarray(self.seed_positions_dimless, dtype=float)
            ),
            "seed_spacing": self.seed_spacing,
            "seed_fits_bounds_directly": self.seed_fits_bounds_directly,
            "seed_jitter": self.seed_jitter,
            "seed_rng_seed": self.seed_rng_seed,
            "alpha_eff": self.alpha_eff,
            "reference_point_dimless": (
                None
                if self.reference_point_dimless is None
                else np.asarray(self.reference_point_dimless, dtype=float)
            ),
            "stage": str(self.stage),
            "optimizer_path": self.optimizer_path,
            "projected_gradient_max_norm": self.projected_gradient_max_norm,
            "distance_moved_from_seed_dimless": self.distance_moved_from_seed_dimless,
            "nfev": self.nfev,
            "njev": self.njev,
            "nit": self.nit,
            "basin_iterations": self.basin_iterations,
            "num_local_attempts": int(self.num_local_attempts),
            "num_basin_attempts": int(self.num_basin_attempts),
            "local_top_energies": (
                None
                if self.local_top_energies is None
                else np.asarray(self.local_top_energies, dtype=float)
            ),
            "minimum_pair_separation": self.minimum_pair_separation,
        }


class _PlanarEquilibriumObjective:
    def __init__(self, sim: Any, num_ions: int, plane: PlaneDefinition):
        self.sim = sim
        self.num_ions = int(num_ions)
        self.plane = plane

    def coords2d_to_positions3d(self, flat_coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(flat_coords, dtype=float).reshape(self.num_ions, 2)
        return self.plane.coords_to_lab(coords)

    def energy(self, flat_coords: np.ndarray) -> float:
        positions_3d = self.coords2d_to_positions3d(flat_coords)
        return float(
            self.sim.get_U_using_polyfit_dimensionless(positions_3d.reshape(-1))
        )

    def gradient(self, flat_coords: np.ndarray) -> np.ndarray:
        positions_3d = self.coords2d_to_positions3d(flat_coords)
        gradient_3d = np.asarray(
            self.sim.get_U_Grad_using_polyfit_dimensionless(positions_3d.reshape(-1)),
            dtype=float,
        ).reshape(self.num_ions, 3)
        gradient_2d = gradient_3d @ self.plane.basis
        return gradient_2d.reshape(-1)


class _Cryo2DCopyObjective:
    def __init__(self, sim: Any, num_ions: int):
        self.sim = sim
        self.num_ions = int(num_ions)
        self.plane = get_default_quartic2d_plane()

    def vector_to_coords(self, flat_coords: np.ndarray) -> np.ndarray:
        vec = np.asarray(flat_coords, dtype=float).reshape(2 * self.num_ions)
        x_coords = vec[: self.num_ions]
        z_coords = vec[self.num_ions :]
        return np.column_stack([x_coords, z_coords])

    def coords_to_vector(self, coords_2d: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords_2d, dtype=float).reshape(self.num_ions, 2)
        return np.concatenate([coords[:, 0], coords[:, 1]])

    def vector_to_positions3d(self, flat_coords: np.ndarray) -> np.ndarray:
        coords = self.vector_to_coords(flat_coords)
        positions = np.zeros((self.num_ions, 3), dtype=float)
        positions[:, 0] = coords[:, 0]
        positions[:, 2] = coords[:, 1]
        return positions

    def energy(self, flat_coords: np.ndarray) -> float:
        positions_3d = self.vector_to_positions3d(flat_coords)
        return float(
            self.sim.get_U_using_polyfit_dimensionless(positions_3d.reshape(-1))
        )

    def gradient(self, flat_coords: np.ndarray) -> np.ndarray:
        positions_3d = self.vector_to_positions3d(flat_coords)
        gradient_3d = np.asarray(
            self.sim.get_U_Grad_using_polyfit_dimensionless(positions_3d.reshape(-1)),
            dtype=float,
        ).reshape(self.num_ions, 3)
        grad_x = gradient_3d[:, 0].copy()
        grad_z = gradient_3d[:, 2].copy()

        # Match the Cryo2D notebook behavior by removing the COM mode
        # separately in the two in-plane directions.
        grad_x -= np.mean(grad_x)
        grad_z -= np.mean(grad_z)
        return np.concatenate([grad_x, grad_z])


def _optimizer_success_flag(result: Any, grad_norm: float) -> bool:
    return bool(getattr(result, "success", False)) or bool(
        grad_norm <= _PROJECTED_GRADIENT_SUCCESS_TOL
    )


def _ordered_cryo2dcopy_attempt(
    objective: _Cryo2DCopyObjective,
    flat_coords: np.ndarray,
    *,
    energy: float,
    result: Any,
    message: str,
    optimizer_name: str,
    stage: str,
    seed_family: str,
    seed_vector: np.ndarray,
    seed_spacing: float,
    seed_jitter: float,
    seed_rng_seed: int,
    basin_iterations: int | None = None,
) -> _OptimizationAttempt:
    coords_2d = objective.vector_to_coords(flat_coords)
    positions_3d = objective.vector_to_positions3d(flat_coords)
    ordered_positions, order = sort_positions_on_plane(positions_3d, objective.plane)
    ordered_coords = coords_2d[order]
    ordered_vector = objective.coords_to_vector(ordered_coords)

    seed_coords = objective.vector_to_coords(seed_vector)
    seed_positions_3d = objective.vector_to_positions3d(seed_vector)
    ordered_seed_positions, seed_order = sort_positions_on_plane(
        seed_positions_3d,
        objective.plane,
    )
    ordered_seed_coords = seed_coords[seed_order]

    final_energy = float(energy)
    if not np.isfinite(final_energy):
        final_energy = float(objective.energy(ordered_vector))

    grad_norm = _max_projected_gradient_norm(objective, ordered_vector)
    return _OptimizationAttempt(
        positions_2d=ordered_coords,
        positions_3d_dimless=ordered_positions,
        energy=final_energy,
        success=_optimizer_success_flag(result, grad_norm),
        message=str(message),
        optimizer_name=str(optimizer_name),
        stage=str(stage),
        seed_family=str(seed_family),
        seed_positions_2d=ordered_seed_coords,
        seed_positions_3d_dimless=ordered_seed_positions,
        seed_spacing=float(seed_spacing),
        seed_fits_bounds_directly=None,
        projected_gradient_max_norm=grad_norm,
        distance_moved_from_seed_dimless=_distance_moved_from_seed_dimless(
            ordered_coords,
            ordered_seed_coords,
        ),
        nfev=getattr(result, "nfev", None),
        njev=getattr(result, "njev", None),
        nit=getattr(result, "nit", getattr(result, "niter", None)),
        basin_iterations=basin_iterations,
    )


def _quartic2d_bounds(num_ions: int) -> list[tuple[float, float]]:
    lower_2d, upper_2d = get_quartic2d_planar_bounds_arrays()
    x_bounds = (float(lower_2d[0]), float(upper_2d[0]))
    z_bounds = (float(lower_2d[1]), float(upper_2d[1]))
    return [x_bounds, z_bounds] * int(num_ions)


def _bounds_arrays(bounds: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    bounds_arr = np.asarray(bounds, dtype=float)
    return bounds_arr[:, 0], bounds_arr[:, 1]


def _ordered_attempt(
    objective: _PlanarEquilibriumObjective,
    plane: PlaneDefinition,
    coords_2d: np.ndarray,
    *,
    energy: float,
    message: str,
    optimizer_name: str,
    stage: str,
    seed_family: str | None,
    seed_positions_2d: np.ndarray | None,
    seed_spacing: float | None,
    seed_fits_bounds_directly: bool | None = None,
    nfev: int | None,
    njev: int | None,
    nit: int | None,
    basin_iterations: int | None = None,
) -> _OptimizationAttempt:
    positions_3d = objective.plane.coords_to_lab(coords_2d)
    ordered_positions, order = sort_positions_on_plane(positions_3d, plane)
    ordered_coords = np.asarray(coords_2d, dtype=float)[order]

    ordered_seed_2d = None
    ordered_seed_3d = None
    if seed_positions_2d is not None:
        seed_positions_3d = objective.plane.coords_to_lab(seed_positions_2d)
        ordered_seed_3d, seed_order = sort_positions_on_plane(seed_positions_3d, plane)
        ordered_seed_2d = np.asarray(seed_positions_2d, dtype=float)[seed_order]

    final_energy = float(energy)
    if not np.isfinite(final_energy):
        final_energy = float(objective.energy(ordered_coords.reshape(-1)))
    projected_gradient_max_norm = _max_projected_gradient_norm(
        objective,
        ordered_coords.reshape(-1),
    )
    distance_moved = _distance_moved_from_seed_dimless(ordered_coords, ordered_seed_2d)
    success = bool(
        np.isfinite(final_energy)
        and projected_gradient_max_norm <= _PROJECTED_GRADIENT_SUCCESS_TOL
    )

    return _OptimizationAttempt(
        positions_2d=ordered_coords,
        positions_3d_dimless=ordered_positions,
        energy=final_energy,
        success=bool(success),
        message=str(message),
        optimizer_name=str(optimizer_name),
        stage=str(stage),
        seed_family=seed_family,
        seed_positions_2d=ordered_seed_2d,
        seed_positions_3d_dimless=ordered_seed_3d,
        seed_spacing=seed_spacing,
        seed_fits_bounds_directly=seed_fits_bounds_directly,
        projected_gradient_max_norm=projected_gradient_max_norm,
        distance_moved_from_seed_dimless=distance_moved,
        nfev=nfev,
        njev=njev,
        nit=nit,
        basin_iterations=basin_iterations,
    )


def _run_local_optimization(
    objective: _PlanarEquilibriumObjective,
    plane: PlaneDefinition,
    bounds: list[tuple[float, float]],
    *,
    seed_positions_2d: np.ndarray,
    seed_family: str,
    seed_spacing: float | None,
    seed_fits_bounds_directly: bool | None = None,
) -> _OptimizationAttempt:
    result, total_nfev, total_njev, total_nit = _run_lbfgsb_with_restarts(
        objective,
        np.asarray(seed_positions_2d, dtype=float).reshape(-1),
        bounds,
        options={
            "gtol": 1.0e-12,
            "ftol": 1.0e-14,
            "maxiter": 24000,
            "maxfun": 280000,
            "maxls": 1200,
            "disp": False,
        },
        max_restarts=2,
    )
    if result is None or getattr(result, "x", None) is None:
        final_coords = np.asarray(seed_positions_2d, dtype=float).reshape(objective.num_ions, 2)
        return _ordered_attempt(
            objective,
            plane,
            final_coords,
            energy=np.inf,
            message="L-BFGS-B returned no iterate",
            optimizer_name="L-BFGS-B",
            stage="local",
            seed_family=seed_family,
            seed_positions_2d=seed_positions_2d,
            seed_spacing=seed_spacing,
            seed_fits_bounds_directly=seed_fits_bounds_directly,
            nfev=total_nfev,
            njev=total_njev,
            nit=total_nit,
        )
    final_coords = np.asarray(result.x, dtype=float).reshape(objective.num_ions, 2)
    return _ordered_attempt(
        objective,
        plane,
        final_coords,
        energy=float(getattr(result, "fun", np.inf)),
        message=str(getattr(result, "message", "")),
        optimizer_name="L-BFGS-B",
        stage="local",
        seed_family=seed_family,
        seed_positions_2d=seed_positions_2d,
        seed_spacing=seed_spacing,
        seed_fits_bounds_directly=seed_fits_bounds_directly,
        nfev=total_nfev,
        njev=total_njev,
        nit=total_nit,
    )


class _BoundedPlanarStep:
    def __init__(
        self,
        lower_2d: np.ndarray,
        upper_2d: np.ndarray,
        *,
        num_ions: int,
        step_scale: float,
        rng: np.random.Generator,
    ):
        self.lower_2d = np.asarray(lower_2d, dtype=float).reshape(2)
        self.upper_2d = np.asarray(upper_2d, dtype=float).reshape(2)
        self.num_ions = int(num_ions)
        self.step_scale = float(step_scale)
        self.rng = rng

    def __call__(self, x: np.ndarray) -> np.ndarray:
        trial = np.asarray(x, dtype=float) + self.rng.normal(
            scale=self.step_scale,
            size=np.asarray(x).shape,
        )
        lower = np.tile(self.lower_2d, self.num_ions)
        upper = np.tile(self.upper_2d, self.num_ions)
        return np.clip(trial, lower, upper)


def _run_basin_hopping(
    objective: _PlanarEquilibriumObjective,
    plane: PlaneDefinition,
    bounds: list[tuple[float, float]],
    *,
    start_attempt: _OptimizationAttempt,
    rng: np.random.Generator,
) -> _OptimizationAttempt:
    lower, upper = _bounds_arrays(bounds)
    stepper = _BoundedPlanarStep(
        lower[:2],
        upper[:2],
        num_ions=objective.num_ions,
        step_scale=0.12 * float(max(start_attempt.seed_spacing or 1.0, 1.0e-12)),
        rng=rng,
    )
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "jac": objective.gradient,
        "bounds": bounds,
        "options": {
            "gtol": 1.0e-12,
            "ftol": 1.0e-14,
            "maxiter": 22000,
            "maxfun": 220000,
            "maxls": 1200,
            "disp": False,
        },
    }
    basin = basinhopping(
        objective.energy,
        np.asarray(start_attempt.positions_2d, dtype=float).reshape(-1),
        minimizer_kwargs=minimizer_kwargs,
        niter=90,
        T=0.8,
        take_step=stepper,
        disp=False,
    )

    local_result = getattr(basin, "lowest_optimization_result", None)
    if local_result is None:
        final_coords = np.asarray(getattr(basin, "x"), dtype=float).reshape(
            objective.num_ions, 2
        )
        return _ordered_attempt(
            objective,
            plane,
            final_coords,
            energy=float(getattr(basin, "fun", np.inf)),
            message="basinhopping completed without local result metadata",
            optimizer_name="basinhopping(L-BFGS-B)",
            stage="basinhopping",
            seed_family=start_attempt.seed_family,
            seed_positions_2d=start_attempt.seed_positions_2d,
            seed_spacing=start_attempt.seed_spacing,
            seed_fits_bounds_directly=start_attempt.seed_fits_bounds_directly,
            nfev=None,
            njev=None,
            nit=None,
            basin_iterations=getattr(basin, "nit", None),
        )

    final_coords = np.asarray(local_result.x, dtype=float).reshape(
        objective.num_ions, 2
    )
    return _ordered_attempt(
        objective,
        plane,
        final_coords,
        energy=float(getattr(local_result, "fun", getattr(basin, "fun", np.inf))),
        message=str(getattr(local_result, "message", "basinhopping finished")),
        optimizer_name="basinhopping(L-BFGS-B)",
        stage="basinhopping",
        seed_family=start_attempt.seed_family,
        seed_positions_2d=start_attempt.seed_positions_2d,
        seed_spacing=start_attempt.seed_spacing,
        seed_fits_bounds_directly=start_attempt.seed_fits_bounds_directly,
        nfev=getattr(local_result, "nfev", None),
        njev=getattr(local_result, "njev", None),
        nit=getattr(local_result, "nit", None),
        basin_iterations=getattr(basin, "nit", None),
    )


def _run_final_polish(
    objective: _PlanarEquilibriumObjective,
    plane: PlaneDefinition,
    bounds: list[tuple[float, float]],
    *,
    start_attempt: _OptimizationAttempt,
) -> _OptimizationAttempt:
    lower, upper = _bounds_arrays(bounds)
    result = minimize(
        objective.energy,
        np.asarray(start_attempt.positions_2d, dtype=float).reshape(-1),
        method="trust-constr",
        jac=objective.gradient,
        hess=BFGS(),
        bounds=Bounds(lower, upper),
        options={
            "gtol": 1.0e-10,
            "xtol": 1.0e-12,
            "barrier_tol": 1.0e-12,
            "maxiter": 2000,
            "verbose": 0,
        },
    )
    if result is None or getattr(result, "x", None) is None:
        final_coords = np.asarray(start_attempt.positions_2d, dtype=float)
        return _ordered_attempt(
            objective,
            plane,
            final_coords,
            energy=float(start_attempt.energy),
            message="final polish returned no iterate",
            optimizer_name="L-BFGS-B(final-polish)",
            stage="final_polish",
            seed_family=start_attempt.seed_family,
            seed_positions_2d=start_attempt.seed_positions_2d,
            seed_spacing=start_attempt.seed_spacing,
            seed_fits_bounds_directly=start_attempt.seed_fits_bounds_directly,
            nfev=None,
            njev=None,
            nit=None,
        )
    final_coords = np.asarray(result.x, dtype=float).reshape(objective.num_ions, 2)
    return _ordered_attempt(
        objective,
        plane,
        final_coords,
        energy=float(getattr(result, "fun", np.inf)),
        message=str(getattr(result, "message", "")),
        optimizer_name="trust-constr(final-polish)",
        stage="final_polish",
        seed_family=start_attempt.seed_family,
        seed_positions_2d=start_attempt.seed_positions_2d,
        seed_spacing=start_attempt.seed_spacing,
        seed_fits_bounds_directly=start_attempt.seed_fits_bounds_directly,
        nfev=getattr(result, "nfev", None),
        njev=getattr(result, "njev", None),
        nit=getattr(result, "nit", getattr(result, "niter", None)),
    )


def solve_cryo2dcopy(
    sim: Any,
    num_ions: int,
) -> EquilibriumSearchResult:
    plane = get_default_quartic2d_plane()
    if hasattr(sim, "_ensure_dc_center_fit"):
        sim._ensure_dc_center_fit(polyfit_deg=4)

    objective = _Cryo2DCopyObjective(sim, num_ions)
    seed_context = build_cryo2dcopy_seed_context(sim)
    seed_spacing = 1.0
    seed_jitter = 0.02
    seed_rng_seed = 11
    seed_family = "cryo_triangular_exact"

    seed = make_cryo2dcopy_triangular_seed(
        num_ions,
        seed_context.alpha_seed,
        spacing=seed_spacing,
        jitter=seed_jitter,
        rng_seed=seed_rng_seed,
    )

    bh_kwargs = {
        "method": "BFGS",
        "jac": objective.gradient,
        "options": {
            "gtol": 1.0e-8,
            "disp": False,
        },
    }
    bh_res = basinhopping(
        objective.energy,
        seed.uv0,
        niter=60,
        T=0.5,
        minimizer_kwargs=bh_kwargs,
        disp=False,
    )
    bh_attempt = _ordered_cryo2dcopy_attempt(
        objective,
        np.asarray(bh_res.x, dtype=float),
        energy=float(getattr(bh_res, "fun", np.inf)),
        result=getattr(bh_res, "lowest_optimization_result", bh_res),
        message="basinhopping(BFGS) completed",
        optimizer_name="basinhopping(BFGS)",
        stage="basinhopping",
        seed_family=seed_family,
        seed_vector=seed.uv0,
        seed_spacing=seed_spacing,
        seed_jitter=seed_jitter,
        seed_rng_seed=seed_rng_seed,
        basin_iterations=getattr(bh_res, "nit", None),
    )

    refine_res = minimize(
        objective.energy,
        np.asarray(bh_res.x, dtype=float),
        jac=objective.gradient,
        method="trust-constr",
        options={
            "gtol": 1.0e-9,
            "verbose": 0,
        },
    )
    final_vector = (
        np.asarray(refine_res.x, dtype=float)
        if getattr(refine_res, "x", None) is not None
        else np.asarray(bh_res.x, dtype=float)
    )
    final_attempt = _ordered_cryo2dcopy_attempt(
        objective,
        final_vector,
        energy=float(getattr(refine_res, "fun", np.inf)),
        result=refine_res,
        message=str(getattr(refine_res, "message", "")),
        optimizer_name="basinhopping(BFGS)->trust-constr",
        stage="final_polish",
        seed_family=seed_family,
        seed_vector=seed.uv0,
        seed_spacing=seed_spacing,
        seed_jitter=seed_jitter,
        seed_rng_seed=seed_rng_seed,
        basin_iterations=getattr(bh_res, "nit", None),
    )

    positions_si = (
        np.asarray(final_attempt.positions_3d_dimless, dtype=float)
        * constants.length_harmonic_approximation
    )
    seed_positions_si = (
        np.asarray(final_attempt.seed_positions_3d_dimless, dtype=float)
        * constants.length_harmonic_approximation
    )
    optimizer_path = "basinhopping(BFGS)->trust-constr"
    return EquilibriumSearchResult(
        positions=positions_si,
        positions_dimless=np.asarray(final_attempt.positions_3d_dimless, dtype=float),
        energy=float(final_attempt.energy),
        success=bool(final_attempt.success),
        message=str(final_attempt.message),
        optimizer_name=optimizer_path,
        minimizer_name=CRYO2DCOPY,
        plane_normal=np.asarray(plane.normal, dtype=float),
        seed_family=seed_family,
        seed_positions=seed_positions_si,
        seed_positions_dimless=np.asarray(
            final_attempt.seed_positions_3d_dimless,
            dtype=float,
        ),
        seed_spacing=seed_spacing,
        seed_fits_bounds_directly=None,
        seed_jitter=seed_jitter,
        seed_rng_seed=seed_rng_seed,
        alpha_eff=seed_context.alpha_seed,
        reference_point_dimless=np.asarray(
            seed_context.reference_point_dimless,
            dtype=float,
        ),
        stage="final_polish",
        optimizer_path=optimizer_path,
        projected_gradient_max_norm=final_attempt.projected_gradient_max_norm,
        distance_moved_from_seed_dimless=final_attempt.distance_moved_from_seed_dimless,
        nfev=getattr(refine_res, "nfev", None),
        njev=getattr(refine_res, "njev", None),
        nit=getattr(refine_res, "nit", getattr(refine_res, "niter", None)),
        basin_iterations=getattr(bh_res, "nit", None),
        num_local_attempts=1,
        num_basin_attempts=1,
        local_top_energies=np.asarray([bh_attempt.energy], dtype=float),
        minimum_pair_separation=_minimum_pair_separation(positions_si),
    )


def solve_quartic2d_101(
    sim: Any,
    num_ions: int,
    *,
    plane: PlaneDefinition | None = None,
) -> EquilibriumSearchResult:
    plane = plane or get_default_quartic2d_plane()
    if hasattr(sim, "_ensure_dc_center_fit"):
        sim._ensure_dc_center_fit(polyfit_deg=4)

    objective = _PlanarEquilibriumObjective(sim, num_ions, plane)
    bounds = _quartic2d_bounds(num_ions)
    rng = np.random.default_rng()
    seed_context = build_quartic2d_seed_context(sim, plane)
    seed_candidates = generate_quartic2d_seed_candidates(
        sim,
        num_ions,
        plane=plane,
        context=seed_context,
        rng=rng,
    )
    if not seed_candidates:
        zero_positions_dimless = np.zeros((int(num_ions), 3), dtype=float)
        return EquilibriumSearchResult(
            positions=zero_positions_dimless * constants.length_harmonic_approximation,
            positions_dimless=zero_positions_dimless,
            energy=float(np.inf),
            success=False,
            message="No Cryo triangular seeds fit within the widened Quartic2D bounds.",
            optimizer_name="seed-generation",
            minimizer_name=QUARTIC2D_101,
            plane_normal=np.asarray(plane.normal, dtype=float),
            seed_family="cryo_triangular",
            seed_positions=None,
            seed_positions_dimless=None,
            seed_spacing=None,
            seed_fits_bounds_directly=False,
            alpha_eff=seed_context.alpha_eff,
            reference_point_dimless=np.asarray(
                seed_context.reference_point_3d_dimless, dtype=float
            ),
            stage="seed_generation",
            projected_gradient_max_norm=None,
            distance_moved_from_seed_dimless=None,
            nfev=None,
            njev=None,
            nit=None,
            basin_iterations=None,
            num_local_attempts=0,
            num_basin_attempts=0,
            local_top_energies=None,
            minimum_pair_separation=None,
        )

    local_attempts = [
        _run_local_optimization(
            objective,
            plane,
            bounds,
            seed_positions_2d=candidate["positions_2d"],
            seed_family=str(candidate["family"]),
            seed_spacing=float(candidate.get("spacing")),
            seed_fits_bounds_directly=bool(candidate.get("seed_fits_bounds_directly", True)),
        )
        for candidate in seed_candidates
    ]
    ranked_local = sorted(local_attempts, key=_attempt_sort_key)
    top_local = ranked_local[: min(3, len(ranked_local))]

    basin_attempts = [
        _run_basin_hopping(objective, plane, bounds, start_attempt=attempt, rng=rng)
        for attempt in top_local
    ]

    candidate_pool = top_local + basin_attempts
    winning_attempt = _best_attempt(candidate_pool or ranked_local)
    polished_attempt = _run_final_polish(
        objective,
        plane,
        bounds,
        start_attempt=winning_attempt,
    )
    winning_attempt = _best_attempt([winning_attempt, polished_attempt])
    positions_si = (
        np.asarray(winning_attempt.positions_3d_dimless, dtype=float)
        * constants.length_harmonic_approximation
    )
    seed_positions_si = None
    if winning_attempt.seed_positions_3d_dimless is not None:
        seed_positions_si = (
            np.asarray(winning_attempt.seed_positions_3d_dimless, dtype=float)
            * constants.length_harmonic_approximation
        )

    local_top_energies = np.asarray(
        [attempt.energy for attempt in ranked_local[: min(3, len(ranked_local))]],
        dtype=float,
    )

    return EquilibriumSearchResult(
        positions=positions_si,
        positions_dimless=np.asarray(winning_attempt.positions_3d_dimless, dtype=float),
        energy=float(winning_attempt.energy),
        success=bool(winning_attempt.success),
        message=str(winning_attempt.message),
        optimizer_name=str(winning_attempt.optimizer_name),
        minimizer_name=QUARTIC2D_101,
        plane_normal=np.asarray(plane.normal, dtype=float),
        seed_family=winning_attempt.seed_family,
        seed_positions=seed_positions_si,
        seed_positions_dimless=winning_attempt.seed_positions_3d_dimless,
        seed_spacing=winning_attempt.seed_spacing,
        seed_fits_bounds_directly=winning_attempt.seed_fits_bounds_directly,
        alpha_eff=seed_context.alpha_eff,
        reference_point_dimless=np.asarray(
            seed_context.reference_point_3d_dimless, dtype=float
        ),
        stage=str(winning_attempt.stage),
        projected_gradient_max_norm=winning_attempt.projected_gradient_max_norm,
        distance_moved_from_seed_dimless=winning_attempt.distance_moved_from_seed_dimless,
        nfev=winning_attempt.nfev,
        njev=winning_attempt.njev,
        nit=winning_attempt.nit,
        basin_iterations=winning_attempt.basin_iterations,
        num_local_attempts=len(local_attempts),
        num_basin_attempts=len(basin_attempts),
        local_top_energies=local_top_energies,
        minimum_pair_separation=_minimum_pair_separation(positions_si),
    )
