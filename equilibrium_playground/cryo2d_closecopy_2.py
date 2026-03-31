from __future__ import annotations

"""Standalone Cryo-inspired solver for the reduced even 2D model.

This module intentionally does not depend on the main simulation stack. It
works directly with manually supplied reduced polynomial coefficients and
dimensionless x-z coordinates.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import basinhopping, minimize

from equilibrium_playground.reduced_model import (
    coulomb_energy,
    pack_xz,
    split_xz,
    total_energy,
    total_gradient,
)


_EPS = 1.0e-12


@dataclass(frozen=True)
class Cryo2DTriangularSeed:
    """One Cryo-style triangular seed in dimensionless x-z coordinates."""

    points_2d: np.ndarray
    x0: np.ndarray
    z0: np.ndarray
    q0: np.ndarray
    spacing: float
    jitter: float
    rng_seed: int | None


@dataclass
class Cryo2DCloseCopy2Result:
    """Result bundle for the standalone reduced-model solver."""

    positions_2d: np.ndarray
    q: np.ndarray
    energy: float
    success: bool
    message: str
    optimizer_path: str
    seed_family: str
    seed_positions_2d: np.ndarray
    seed_q: np.ndarray
    seed_spacing: float
    seed_jitter: float
    seed_rng_seed: int | None
    alpha_seed: float
    coefficients: dict[str, float]
    projected_gradient_max_norm: float
    seed_to_final_rms_move: float
    nfev: int | None = None
    njev: int | None = None
    nit: int | None = None
    basin_iterations: int | None = None
    basinhopping_success: bool | None = None
    basinhopping_message: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "minimizer_name": "Cryo2d_closecopy_2_playground",
            "optimizer_name": self.optimizer_path,
            "optimizer_path": self.optimizer_path,
            "seed_family": self.seed_family,
            "energy": float(self.energy),
            "success": bool(self.success),
            "message": str(self.message),
            "seed_spacing": float(self.seed_spacing),
            "seed_jitter": float(self.seed_jitter),
            "seed_rng_seed": self.seed_rng_seed,
            "alpha_eff": float(self.alpha_seed),
            "projected_gradient_max_norm": float(self.projected_gradient_max_norm),
            "distance_moved_from_seed_dimless": float(self.seed_to_final_rms_move),
            "a20": float(self.coefficients["a20"]),
            "a02": float(self.coefficients["a02"]),
            "a40": float(self.coefficients["a40"]),
            "a22": float(self.coefficients["a22"]),
            "a04": float(self.coefficients["a04"]),
        }


def _hex_shell_population(radius: int) -> int:
    r_val = int(radius)
    if r_val < 0:
        raise ValueError("radius must be nonnegative")
    return 1 + 3 * r_val * (r_val + 1)


def _minimal_hex_shell_radius(num_ions: int) -> int:
    n = int(num_ions)
    if n <= 0:
        raise ValueError("num_ions must be positive")
    radius = 0
    while _hex_shell_population(radius) < n:
        radius += 1
    return radius


def _generate_axial_hex_sites(radius: int) -> np.ndarray:
    r_max = int(radius)
    sites = []
    for q_coord in range(-r_max, r_max + 1):
        for r_coord in range(-r_max, r_max + 1):
            s_coord = -q_coord - r_coord
            if max(abs(q_coord), abs(r_coord), abs(s_coord)) <= r_max:
                sites.append((q_coord, r_coord))
    return np.asarray(sites, dtype=int)


def _projected_gradient_max_norm(grad_q: np.ndarray) -> float:
    grad = np.asarray(grad_q, dtype=float).reshape(-1)
    if grad.size == 0:
        return 0.0
    return float(np.max(np.abs(grad)))


def _seed_to_final_rms_move(final_q: np.ndarray, seed_q: np.ndarray) -> float:
    final_x, final_z = split_xz(final_q)
    seed_x, seed_z = split_xz(seed_q)
    dx = final_x - seed_x
    dz = final_z - seed_z
    return float(np.sqrt(np.mean(dx * dx + dz * dz)))


def _sort_positions_xz(points_2d: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    return pts[order]


def _alpha_from_quadratic_coefficients(a20: float, a02: float) -> float:
    curv_x = max(2.0 * float(a20), _EPS)
    curv_z = max(2.0 * float(a02), _EPS)
    return float(np.sqrt(curv_z / curv_x))


def make_triangular_seed(
    num_ions: int,
    alpha_yx: float,
    spacing: float = 1.0,
    *,
    jitter: float = 0.0,
    rng_seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> Cryo2DTriangularSeed:
    """Build a Cryo-style triangular seed in dimensionless x-z coordinates.

    The geometry follows the notebook helper closely:
    - minimal hex shell
    - axial coordinates
    - `x = spacing * (q + r/2)`
    - `z = spacing * (sqrt(3)/2) * r / alpha_yx`
    - keep the N closest sites in the trap metric
    - optional tiny jitter
    - center the center of mass
    """

    rng_obj = np.random.default_rng(rng_seed) if rng is None else rng
    radius = _minimal_hex_shell_radius(num_ions)
    pts = []
    alpha = float(max(alpha_yx, _EPS))
    spacing_value = float(spacing)
    for q_coord, r_coord in _generate_axial_hex_sites(radius):
        x_coord = spacing_value * (q_coord + 0.5 * r_coord)
        z_coord = spacing_value * (np.sqrt(3.0) / 2.0) * r_coord / alpha
        pts.append((x_coord, z_coord))
    pts_arr = np.asarray(pts, dtype=float)
    trap_radius = np.sqrt(pts_arr[:, 0] ** 2 + (alpha * pts_arr[:, 1]) ** 2)
    keep = np.argsort(trap_radius, kind="stable")[: int(num_ions)]
    pts_arr = pts_arr[keep].copy()
    if jitter > 0.0:
        pts_arr += float(jitter) * spacing_value * rng_obj.standard_normal(pts_arr.shape)
    pts_arr -= pts_arr.mean(axis=0, keepdims=True)
    x0 = pts_arr[:, 0].copy()
    z0 = pts_arr[:, 1].copy()
    return Cryo2DTriangularSeed(
        points_2d=pts_arr,
        x0=x0,
        z0=z0,
        q0=pack_xz(x0, z0),
        spacing=spacing_value,
        jitter=float(jitter),
        rng_seed=rng_seed,
    )


def solve_cryo2d_closecopy_2(
    num_ions: int,
    a20: float,
    a02: float,
    a40: float,
    a22: float,
    a04: float,
    *,
    spacing: float = 1.0,
    jitter: float = 0.02,
    rng_seed: int | None = None,
    basinhopping_niter: int = 60,
    basin_temperature: float = 0.5,
    bfgs_gtol: float = 1.0e-10,
    bfgs_maxiter: int = 20000,
    trust_gtol: float = 1.0e-11,
    trust_xtol: float = 1.0e-12,
    trust_barrier_tol: float = 1.0e-12,
    trust_maxiter: int = 4000,
) -> Cryo2DCloseCopy2Result:
    """Run the standalone Cryo-inspired reduced-model minimizer.

    All inputs and outputs are dimensionless. Randomness remains unfixed by
    default. If a caller supplies `rng_seed`, that explicit choice is honored.
    """

    coefficients = {
        "a20": float(a20),
        "a02": float(a02),
        "a40": float(a40),
        "a22": float(a22),
        "a04": float(a04),
    }
    alpha_seed = _alpha_from_quadratic_coefficients(a20, a02)
    seed = make_triangular_seed(
        int(num_ions),
        alpha_seed,
        spacing=spacing,
        jitter=jitter,
        rng_seed=rng_seed,
    )

    energy_fn = lambda q: total_energy(q, **coefficients)
    gradient_fn = lambda q: total_gradient(q, **coefficients)

    bh_kwargs = {
        "method": "BFGS",
        "jac": gradient_fn,
        "options": {
            "gtol": float(bfgs_gtol),
            "disp": False,
            "maxiter": int(bfgs_maxiter),
        },
    }
    bh_res = basinhopping(
        energy_fn,
        seed.q0,
        minimizer_kwargs=bh_kwargs,
        niter=int(basinhopping_niter),
        T=float(basin_temperature),
        disp=False,
        seed=rng_seed,
    )

    refine_res = minimize(
        energy_fn,
        np.asarray(bh_res.x, dtype=float),
        jac=gradient_fn,
        method="trust-constr",
        options={
            "gtol": float(trust_gtol),
            "xtol": float(trust_xtol),
            "barrier_tol": float(trust_barrier_tol),
            "maxiter": int(trust_maxiter),
            "verbose": 0,
        },
    )

    final_q = (
        np.asarray(refine_res.x, dtype=float)
        if getattr(refine_res, "x", None) is not None
        else np.asarray(bh_res.x, dtype=float)
    )
    final_points = np.column_stack(split_xz(final_q))
    seed_points = _sort_positions_xz(seed.points_2d)
    final_points = _sort_positions_xz(final_points)
    seed_q_sorted = pack_xz(seed_points[:, 0], seed_points[:, 1])
    final_q_sorted = pack_xz(final_points[:, 0], final_points[:, 1])
    final_grad = gradient_fn(final_q_sorted)

    return Cryo2DCloseCopy2Result(
        positions_2d=final_points,
        q=final_q_sorted,
        energy=float(getattr(refine_res, "fun", energy_fn(final_q_sorted))),
        success=bool(getattr(refine_res, "success", False)),
        message=str(getattr(refine_res, "message", "")),
        optimizer_path="basinhopping(BFGS)->trust-constr",
        seed_family="cryo_triangular",
        seed_positions_2d=seed_points,
        seed_q=seed_q_sorted,
        seed_spacing=float(spacing),
        seed_jitter=float(jitter),
        seed_rng_seed=rng_seed,
        alpha_seed=alpha_seed,
        coefficients=coefficients,
        projected_gradient_max_norm=_projected_gradient_max_norm(final_grad),
        seed_to_final_rms_move=_seed_to_final_rms_move(final_q_sorted, seed_q_sorted),
        nfev=getattr(refine_res, "nfev", None),
        njev=getattr(refine_res, "njev", None),
        nit=getattr(refine_res, "nit", None),
        basin_iterations=int(basinhopping_niter),
        basinhopping_success=bool(getattr(bh_res, "lowest_optimization_result", None) is not None),
        basinhopping_message=str(getattr(bh_res, "message", "")),
    )
