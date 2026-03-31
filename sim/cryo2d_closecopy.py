from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, basinhopping, minimize

import constants


CRYO2D_CLOSECOPY = "Cryo2d_closecopy"
_BOUNDS_SCALE = 3.0
_EPS = 1.0e-12
_PLANE_NORMAL = np.array([0.0, 1.0, 0.0], dtype=float)
_OPTIMIZER_PATH = "basinhopping(BFGS)->trust-constr"

# Exactly 3 Cryo-style seeds: same triangular helper, varying only
# spacing.
DEFAULT_CLOSECOPY_SEEDS: tuple[tuple[float, int], ...] = (
    (0.90, 11),
    (1.00, 11),
    (1.10, 11),
)


@dataclass(frozen=True)
class Cryo2DTriangularSeed:
    points_2d: np.ndarray
    u0: np.ndarray
    v0: np.ndarray
    uv0: np.ndarray
    spacing: float
    jitter: float
    rng_seed: int


@dataclass(frozen=True)
class Cryo2DCloseCopyContext:
    center_xz_dimless: np.ndarray
    center_3d_dimless: np.ndarray
    curvature_x: float
    curvature_z: float
    alpha_seed: float
    lower_local: np.ndarray
    upper_local: np.ndarray


@dataclass
class _CloseCopyAttempt:
    positions_dimless: np.ndarray
    positions_local_dimless: np.ndarray
    positions: np.ndarray
    energy: float
    success: bool
    message: str
    seed_positions_dimless: np.ndarray
    seed_positions_local_dimless: np.ndarray
    seed_positions: np.ndarray
    seed_spacing: float
    seed_jitter: float
    seed_rng_seed: int
    projected_gradient_max_norm: float
    distance_moved_from_seed_dimless: float
    nfev: int | None
    njev: int | None
    nit: int | None
    basin_iterations: int | None


@dataclass
class Cryo2DCloseCopyResult:
    positions: np.ndarray
    positions_dimless: np.ndarray
    positions_local_dimless: np.ndarray
    energy: float
    success: bool
    message: str
    optimizer_name: str
    minimizer_name: str
    plane_normal: np.ndarray
    seed_family: str
    seed_positions: np.ndarray
    seed_positions_dimless: np.ndarray
    seed_positions_local_dimless: np.ndarray
    seed_spacing: float
    seed_jitter: float
    seed_rng_seed: int
    alpha_eff: float
    reference_point_dimless: np.ndarray
    stage: str
    optimizer_path: str
    projected_gradient_max_norm: float
    distance_moved_from_seed_dimless: float
    nfev: int | None = None
    njev: int | None = None
    nit: int | None = None
    basin_iterations: int | None = None
    num_seed_attempts: int = 0
    minimum_pair_separation: float | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "positions": np.asarray(self.positions, dtype=float),
            "positions_dimless": np.asarray(self.positions_dimless, dtype=float),
            "positions_local_dimless": np.asarray(
                self.positions_local_dimless,
                dtype=float,
            ),
            "energy": float(self.energy),
            "success": bool(self.success),
            "message": str(self.message),
            "optimizer_name": str(self.optimizer_name),
            "optimizer_path": str(self.optimizer_path),
            "minimizer_name": str(self.minimizer_name),
            "plane_normal": np.asarray(self.plane_normal, dtype=float),
            "seed_family": str(self.seed_family),
            "seed_positions": np.asarray(self.seed_positions, dtype=float),
            "seed_positions_dimless": np.asarray(
                self.seed_positions_dimless,
                dtype=float,
            ),
            "seed_positions_local_dimless": np.asarray(
                self.seed_positions_local_dimless,
                dtype=float,
            ),
            "seed_spacing": float(self.seed_spacing),
            "seed_jitter": float(self.seed_jitter),
            "seed_rng_seed": int(self.seed_rng_seed),
            "alpha_eff": float(self.alpha_eff),
            "reference_point_dimless": np.asarray(
                self.reference_point_dimless,
                dtype=float,
            ),
            "stage": str(self.stage),
            "projected_gradient_max_norm": float(self.projected_gradient_max_norm),
            "distance_moved_from_seed_dimless": float(
                self.distance_moved_from_seed_dimless
            ),
            "nfev": self.nfev,
            "njev": self.njev,
            "nit": self.nit,
            "basin_iterations": self.basin_iterations,
            "num_seed_attempts": int(self.num_seed_attempts),
            "minimum_pair_separation": self.minimum_pair_separation,
        }


def _expanded_bounds_xz() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array(
        [
            -_BOUNDS_SCALE * constants.center_x_bounds,
            -_BOUNDS_SCALE * constants.center_z_bounds,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            _BOUNDS_SCALE * constants.center_x_bounds,
            _BOUNDS_SCALE * constants.center_z_bounds,
        ],
        dtype=float,
    )
    return lower, upper


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


def _sort_positions_xz(positions_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(positions_3d, dtype=float).reshape(-1, 3)
    order = np.lexsort((pts[:, 2], pts[:, 0]))
    return pts[order], order


def _local_uv_to_lab_positions(
    uv: np.ndarray,
    center_xz_dimless: np.ndarray,
) -> np.ndarray:
    vec = np.asarray(uv, dtype=float).reshape(-1)
    num_ions = len(vec) // 2
    u = vec[:num_ions]
    v = vec[num_ions:]
    center = np.asarray(center_xz_dimless, dtype=float).reshape(2)
    positions = np.zeros((num_ions, 3), dtype=float)
    positions[:, 0] = center[0] + u
    positions[:, 2] = center[1] + v
    return positions


def _positions_to_local_uv(
    positions_3d: np.ndarray,
    center_xz_dimless: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(positions_3d, dtype=float).reshape(-1, 3)
    center = np.asarray(center_xz_dimless, dtype=float).reshape(2)
    u = pts[:, 0] - center[0]
    v = pts[:, 2] - center[1]
    return np.concatenate([u, v])


def _projected_gradient_max_norm(
    grad_uv: np.ndarray,
) -> float:
    grad = np.asarray(grad_uv, dtype=float).reshape(-1)
    if grad.size == 0:
        return 0.0
    return float(np.max(np.abs(grad)))


def _distance_moved_from_seed_dimless(
    final_local_uv: np.ndarray,
    seed_local_uv: np.ndarray,
) -> float:
    final_vec = np.asarray(final_local_uv, dtype=float).reshape(-1)
    seed_vec = np.asarray(seed_local_uv, dtype=float).reshape(-1)
    num_ions = len(final_vec) // 2
    final_u = final_vec[:num_ions]
    final_v = final_vec[num_ions:]
    seed_u = seed_vec[:num_ions]
    seed_v = seed_vec[num_ions:]
    delta2 = (final_u - seed_u) ** 2 + (final_v - seed_v) ** 2
    return float(np.sqrt(np.mean(delta2)))


def _attempt_sort_key(attempt: _CloseCopyAttempt) -> tuple[int, float]:
    return (
        0 if attempt.success and np.isfinite(attempt.energy) else 1,
        attempt.energy if np.isfinite(attempt.energy) else np.inf,
    )


def _build_local_bounds_vectors(
    num_ions: int,
    center_xz_dimless: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower_lab, upper_lab = _expanded_bounds_xz()
    center = np.asarray(center_xz_dimless, dtype=float).reshape(2)
    lower_local_2d = lower_lab - center
    upper_local_2d = upper_lab - center
    lower = np.concatenate(
        [
            np.full(int(num_ions), lower_local_2d[0], dtype=float),
            np.full(int(num_ions), lower_local_2d[1], dtype=float),
        ]
    )
    upper = np.concatenate(
        [
            np.full(int(num_ions), upper_local_2d[0], dtype=float),
            np.full(int(num_ions), upper_local_2d[1], dtype=float),
        ]
    )
    return lower, upper


def _clip_to_local_bounds(
    uv: np.ndarray,
    lower_local: np.ndarray,
    upper_local: np.ndarray,
) -> np.ndarray:
    vec = np.asarray(uv, dtype=float).reshape(-1)
    lower = np.asarray(lower_local, dtype=float).reshape(-1)
    upper = np.asarray(upper_local, dtype=float).reshape(-1)
    return np.clip(vec, lower, upper)


def _seed_alpha_from_curvatures(
    sim: Any,
    center_3d_dimless: np.ndarray,
) -> tuple[float, float, float]:
    point_si = (
        np.asarray(center_3d_dimless, dtype=float)
        * constants.length_harmonic_approximation
    )
    hessian_si = np.asarray(
        sim.evaluate_center_poly_2ndderivatives(*point_si),
        dtype=float,
    ).reshape(3, 3)
    curvature_x = float(max(abs(hessian_si[0, 0]), _EPS))
    curvature_z = float(max(abs(hessian_si[2, 2]), _EPS))
    alpha_seed = float(np.sqrt(curvature_z / curvature_x))
    return curvature_x, curvature_z, alpha_seed


def find_1ion_planar_minimum(sim: Any) -> Cryo2DCloseCopyContext:
    if hasattr(sim, "_ensure_dc_center_fit"):
        sim._ensure_dc_center_fit(polyfit_deg=4)

    lower, upper = _expanded_bounds_xz()

    def trap_energy(xz: np.ndarray) -> float:
        positions = np.array([[xz[0], 0.0, xz[1]]], dtype=float)
        return float(sim.get_U_using_polyfit_dimensionless(positions.reshape(-1)))

    def trap_gradient(xz: np.ndarray) -> np.ndarray:
        positions = np.array([[xz[0], 0.0, xz[1]]], dtype=float)
        grad3 = np.asarray(
            sim.get_U_Grad_using_polyfit_dimensionless(positions.reshape(-1)),
            dtype=float,
        ).reshape(1, 3)[0]
        return np.array([grad3[0], grad3[2]], dtype=float)

    result = minimize(
        trap_energy,
        np.zeros(2, dtype=float),
        jac=trap_gradient,
        method="trust-constr",
        bounds=Bounds(lower, upper),
        options={
            "gtol": 1.0e-12,
            "xtol": 1.0e-12,
            "barrier_tol": 1.0e-12,
            "maxiter": 1000,
            "verbose": 0,
        },
    )
    center_xz = (
        np.asarray(result.x, dtype=float).reshape(2)
        if getattr(result, "x", None) is not None
        else np.zeros(2, dtype=float)
    )
    center_3d = np.array([center_xz[0], 0.0, center_xz[1]], dtype=float)
    curvature_x, curvature_z, alpha_seed = _seed_alpha_from_curvatures(sim, center_3d)
    lower_local, upper_local = _build_local_bounds_vectors(1, center_xz)
    return Cryo2DCloseCopyContext(
        center_xz_dimless=center_xz,
        center_3d_dimless=center_3d,
        curvature_x=curvature_x,
        curvature_z=curvature_z,
        alpha_seed=alpha_seed,
        lower_local=lower_local[:2].copy(),
        upper_local=upper_local[:2].copy(),
    )


def U2D(
    uv: np.ndarray,
    sim: Any,
    center_xz_dimless: np.ndarray,
) -> float:
    positions = _local_uv_to_lab_positions(uv, center_xz_dimless)
    return float(sim.get_U_using_polyfit_dimensionless(positions.reshape(-1)))


def grad_U2D(
    uv: np.ndarray,
    sim: Any,
    center_xz_dimless: np.ndarray,
) -> np.ndarray:
    positions = _local_uv_to_lab_positions(uv, center_xz_dimless)
    gradient_3d = np.asarray(
        sim.get_U_Grad_using_polyfit_dimensionless(positions.reshape(-1)),
        dtype=float,
    ).reshape(len(positions), 3)
    grad_x = gradient_3d[:, 0].copy()
    grad_z = gradient_3d[:, 2].copy()
    grad_x -= np.mean(grad_x)
    grad_z -= np.mean(grad_z)
    return np.concatenate([grad_x, grad_z])


def make_triangular_seed(
    num_ions: int,
    alpha_yx: float,
    spacing: float = 1.0,
    *,
    jitter: float = 0.0,
    rng_seed: int = 0,
) -> Cryo2DTriangularSeed:
    rng = np.random.default_rng(int(rng_seed))
    radius = _minimal_hex_shell_radius(num_ions)
    pts = []
    for q_coord, r_coord in _generate_axial_hex_sites(radius):
        x_coord = float(spacing) * (q_coord + 0.5 * r_coord)
        z_coord = (
            float(spacing) * (np.sqrt(3.0) / 2.0) * r_coord / float(max(alpha_yx, _EPS))
        )
        pts.append((x_coord, z_coord))
    pts_arr = np.asarray(pts, dtype=float)
    trap_radius = np.sqrt(pts_arr[:, 0] ** 2 + (float(alpha_yx) * pts_arr[:, 1]) ** 2)
    keep = np.argsort(trap_radius, kind="stable")[: int(num_ions)]
    pts_arr = pts_arr[keep].copy()
    if jitter > 0.0:
        pts_arr += float(jitter) * float(spacing) * rng.standard_normal(pts_arr.shape)
    pts_arr -= pts_arr.mean(axis=0, keepdims=True)
    u0 = pts_arr[:, 0].copy()
    v0 = pts_arr[:, 1].copy()
    uv0 = np.concatenate([u0, v0])
    return Cryo2DTriangularSeed(
        points_2d=pts_arr,
        u0=u0,
        v0=v0,
        uv0=uv0,
        spacing=float(spacing),
        jitter=float(jitter),
        rng_seed=int(rng_seed),
    )


def _run_single_seed(
    sim: Any,
    num_ions: int,
    context: Cryo2DCloseCopyContext,
    *,
    spacing: float,
    rng_seed: int,
    jitter: float,
) -> _CloseCopyAttempt:
    seed = make_triangular_seed(
        num_ions,
        context.alpha_seed,
        spacing=spacing,
        jitter=jitter,
        rng_seed=rng_seed,
    )

    energy_fn = lambda x: U2D(x, sim, context.center_xz_dimless)
    gradient_fn = lambda x: grad_U2D(x, sim, context.center_xz_dimless)

    bh_kwargs = {
        "method": "BFGS",
        "jac": gradient_fn,
        "options": {
            "gtol": 1.0e-10,
            "disp": False,
            "maxiter": 20000,
        },
    }
    bh_res = basinhopping(
        energy_fn,
        seed.uv0,
        minimizer_kwargs=bh_kwargs,
        niter=60,
        T=0.5,
        disp=False,
        seed=int(rng_seed),
    )

    lower_local, upper_local = _build_local_bounds_vectors(
        num_ions, context.center_xz_dimless
    )
    refine_start = _clip_to_local_bounds(
        np.asarray(bh_res.x, dtype=float),
        lower_local,
        upper_local,
    )
    refine_res = minimize(
        energy_fn,
        refine_start,
        jac=gradient_fn,
        method="trust-constr",
        bounds=Bounds(lower_local, upper_local),
        options={
            "gtol": 1.0e-11,
            "xtol": 1.0e-12,
            "barrier_tol": 1.0e-12,
            "maxiter": 4000,
            "verbose": 0,
        },
    )
    final_uv = (
        np.asarray(refine_res.x, dtype=float)
        if getattr(refine_res, "x", None) is not None
        else refine_start
    )
    final_positions = _local_uv_to_lab_positions(final_uv, context.center_xz_dimless)
    sorted_positions, order = _sort_positions_xz(final_positions)
    sorted_final_uv = _positions_to_local_uv(
        sorted_positions, context.center_xz_dimless
    )

    seed_positions = _local_uv_to_lab_positions(seed.uv0, context.center_xz_dimless)
    sorted_seed_positions, seed_order = _sort_positions_xz(seed_positions)
    sorted_seed_uv = _positions_to_local_uv(
        sorted_seed_positions,
        context.center_xz_dimless,
    )

    grad_norm = _projected_gradient_max_norm(gradient_fn(sorted_final_uv))
    success = bool(getattr(refine_res, "success", False)) or bool(grad_norm <= 1.0e-8)
    positions_si = sorted_positions * constants.length_harmonic_approximation
    seed_positions_si = sorted_seed_positions * constants.length_harmonic_approximation
    energy = float(getattr(refine_res, "fun", energy_fn(sorted_final_uv)))
    return _CloseCopyAttempt(
        positions_dimless=sorted_positions,
        positions_local_dimless=np.column_stack(
            [
                sorted_final_uv[:num_ions],
                sorted_final_uv[num_ions:],
            ]
        ),
        positions=positions_si,
        energy=energy,
        success=success,
        message=str(getattr(refine_res, "message", "")),
        seed_positions_dimless=sorted_seed_positions,
        seed_positions_local_dimless=np.column_stack(
            [
                sorted_seed_uv[:num_ions],
                sorted_seed_uv[num_ions:],
            ]
        ),
        seed_positions=seed_positions_si,
        seed_spacing=float(spacing),
        seed_jitter=float(jitter),
        seed_rng_seed=int(rng_seed),
        projected_gradient_max_norm=grad_norm,
        distance_moved_from_seed_dimless=_distance_moved_from_seed_dimless(
            sorted_final_uv,
            sorted_seed_uv,
        ),
        nfev=getattr(refine_res, "nfev", None),
        njev=getattr(refine_res, "njev", None),
        nit=getattr(refine_res, "nit", getattr(refine_res, "niter", None)),
        basin_iterations=getattr(bh_res, "nit", None),
    )


def solve_cryo2d_closecopy(
    sim: Any,
    num_ions: int,
) -> Cryo2DCloseCopyResult:
    if hasattr(sim, "_ensure_dc_center_fit"):
        sim._ensure_dc_center_fit(polyfit_deg=4)

    context = find_1ion_planar_minimum(sim)
    attempts: list[_CloseCopyAttempt] = []
    for spacing, rng_seed in DEFAULT_CLOSECOPY_SEEDS:
        try:
            attempts.append(
                _run_single_seed(
                    sim,
                    int(num_ions),
                    context,
                    spacing=float(spacing),
                    rng_seed=int(rng_seed),
                    jitter=0.02,
                )
            )
        except Exception as exc:
            attempts.append(
                _CloseCopyAttempt(
                    positions_dimless=np.zeros((int(num_ions), 3), dtype=float),
                    positions_local_dimless=np.zeros((int(num_ions), 2), dtype=float),
                    positions=np.zeros((int(num_ions), 3), dtype=float),
                    energy=np.inf,
                    success=False,
                    message=str(exc),
                    seed_positions_dimless=np.zeros((int(num_ions), 3), dtype=float),
                    seed_positions_local_dimless=np.zeros(
                        (int(num_ions), 2), dtype=float
                    ),
                    seed_positions=np.zeros((int(num_ions), 3), dtype=float),
                    seed_spacing=float(spacing),
                    seed_jitter=0.02,
                    seed_rng_seed=int(rng_seed),
                    projected_gradient_max_norm=np.inf,
                    distance_moved_from_seed_dimless=np.inf,
                    nfev=None,
                    njev=None,
                    nit=None,
                    basin_iterations=None,
                )
            )

    best_attempt = min(attempts, key=_attempt_sort_key)
    return Cryo2DCloseCopyResult(
        positions=np.asarray(best_attempt.positions, dtype=float),
        positions_dimless=np.asarray(best_attempt.positions_dimless, dtype=float),
        positions_local_dimless=np.asarray(
            best_attempt.positions_local_dimless,
            dtype=float,
        ),
        energy=float(best_attempt.energy),
        success=bool(best_attempt.success),
        message=str(best_attempt.message),
        optimizer_name=_OPTIMIZER_PATH,
        minimizer_name=CRYO2D_CLOSECOPY,
        plane_normal=_PLANE_NORMAL.copy(),
        seed_family="cryo_triangular",
        seed_positions=np.asarray(best_attempt.seed_positions, dtype=float),
        seed_positions_dimless=np.asarray(
            best_attempt.seed_positions_dimless,
            dtype=float,
        ),
        seed_positions_local_dimless=np.asarray(
            best_attempt.seed_positions_local_dimless,
            dtype=float,
        ),
        seed_spacing=float(best_attempt.seed_spacing),
        seed_jitter=float(best_attempt.seed_jitter),
        seed_rng_seed=int(best_attempt.seed_rng_seed),
        alpha_eff=float(context.alpha_seed),
        reference_point_dimless=np.asarray(context.center_3d_dimless, dtype=float),
        stage="final_polish",
        optimizer_path=_OPTIMIZER_PATH,
        projected_gradient_max_norm=float(best_attempt.projected_gradient_max_norm),
        distance_moved_from_seed_dimless=float(
            best_attempt.distance_moved_from_seed_dimless
        ),
        nfev=best_attempt.nfev,
        njev=best_attempt.njev,
        nit=best_attempt.nit,
        basin_iterations=best_attempt.basin_iterations,
        num_seed_attempts=len(DEFAULT_CLOSECOPY_SEEDS),
        minimum_pair_separation=_minimum_pair_separation(best_attempt.positions),
    )
