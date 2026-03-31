from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

import constants


def _as_unit_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = np.linalg.norm(arr)
    if norm <= 0.0:
        raise ValueError("Plane normal must be nonzero.")
    return arr / norm


@dataclass(frozen=True)
class PlaneDefinition:
    normal: np.ndarray
    basis: np.ndarray
    origin: np.ndarray

    @classmethod
    def from_normal(
        cls,
        normal: np.ndarray,
        *,
        origin: np.ndarray | None = None,
        axis_hint: np.ndarray | None = None,
    ) -> "PlaneDefinition":
        n_hat = _as_unit_vector(normal)
        origin_arr = (
            np.zeros(3, dtype=float)
            if origin is None
            else np.asarray(origin, dtype=float).reshape(3)
        )

        if np.allclose(n_hat, np.array([0.0, 1.0, 0.0]), atol=1.0e-12):
            basis = np.column_stack(
                [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            )
            return cls(normal=n_hat, basis=basis, origin=origin_arr)
        if np.allclose(n_hat, np.array([0.0, -1.0, 0.0]), atol=1.0e-12):
            basis = np.column_stack(
                [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])]
            )
            return cls(normal=n_hat, basis=basis, origin=origin_arr)

        hint = (
            np.asarray(axis_hint, dtype=float).reshape(3)
            if axis_hint is not None
            else np.array([1.0, 0.0, 0.0], dtype=float)
        )
        u_vec = hint - np.dot(hint, n_hat) * n_hat
        if np.linalg.norm(u_vec) <= 1.0e-12:
            candidates = np.eye(3, dtype=float)
            best = min(candidates, key=lambda candidate: abs(np.dot(candidate, n_hat)))
            u_vec = best - np.dot(best, n_hat) * n_hat
        u_hat = _as_unit_vector(u_vec)
        v_hat = _as_unit_vector(np.cross(n_hat, u_hat))
        basis = np.column_stack([u_hat, v_hat])
        return cls(normal=n_hat, basis=basis, origin=origin_arr)

    def coords_to_lab(self, coords_2d: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords_2d, dtype=float)
        return self.origin + coords @ self.basis.T

    def lab_to_coords(self, positions_3d: np.ndarray) -> np.ndarray:
        pts = np.asarray(positions_3d, dtype=float) - self.origin
        return pts @ self.basis


@dataclass(frozen=True)
class PlanarSeedContext:
    reference_point_2d: np.ndarray
    reference_point_3d_dimless: np.ndarray
    curvature_x: float
    curvature_z: float
    alpha_eff: float
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class CryoTriangularSeedLocal:
    points_2d: np.ndarray
    u0: np.ndarray
    v0: np.ndarray
    uv0: np.ndarray


@dataclass(frozen=True)
class Cryo2DCopySeedContext:
    reference_point_dimless: np.ndarray
    curvature_x: float
    curvature_z: float
    alpha_seed: float


QUARTIC2D_BOUNDS_SCALE = 3.0


def _resolve_alpha_yx(
    *,
    alpha_yx: float | None = None,
    alpha_eff: float | None = None,
) -> float:
    if alpha_yx is None and alpha_eff is None:
        raise TypeError("alpha_yx or alpha_eff must be provided.")
    if alpha_yx is not None and alpha_eff is not None and not np.isclose(
        float(alpha_yx),
        float(alpha_eff),
        atol=1.0e-12,
        rtol=1.0e-12,
    ):
        raise ValueError("alpha_yx and alpha_eff must agree when both are provided.")
    return float(alpha_yx if alpha_yx is not None else alpha_eff)


def get_default_quartic2d_plane() -> PlaneDefinition:
    return PlaneDefinition.from_normal(np.array([0.0, 1.0, 0.0], dtype=float))


def get_quartic2d_planar_bounds_arrays(
    *,
    scale: float = QUARTIC2D_BOUNDS_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    scale_val = float(scale)
    lower = np.array(
        [
            -scale_val * constants.center_x_bounds,
            -scale_val * constants.center_z_bounds,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            scale_val * constants.center_x_bounds,
            scale_val * constants.center_z_bounds,
        ],
        dtype=float,
    )
    return lower, upper


def project_hessian_to_plane(
    hessian_3d: np.ndarray,
    plane: PlaneDefinition,
) -> np.ndarray:
    hessian = np.asarray(hessian_3d, dtype=float).reshape(3, 3)
    return plane.basis.T @ hessian @ plane.basis


def sort_positions_on_plane(
    positions_3d: np.ndarray,
    plane: PlaneDefinition,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(positions_3d, dtype=float).reshape(-1, 3)
    coords = plane.lab_to_coords(pts)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return pts[order], order


def hex_shell_population(radius: int) -> int:
    radius_int = int(radius)
    if radius_int < 0:
        raise ValueError("Hex-shell radius must be nonnegative.")
    return 1 + 3 * radius_int * (radius_int + 1)


def minimal_hex_shell_radius(num_ions: int) -> int:
    n = int(num_ions)
    if n <= 0:
        raise ValueError("num_ions must be positive.")
    radius = 0
    while hex_shell_population(radius) < n:
        radius += 1
    return radius


def generate_axial_hex_sites(radius: int) -> np.ndarray:
    r_max = int(radius)
    if r_max < 0:
        raise ValueError("Hex-shell radius must be nonnegative.")

    sites = []
    for q_coord in range(-r_max, r_max + 1):
        for r_coord in range(-r_max, r_max + 1):
            s_coord = -q_coord - r_coord
            if max(abs(q_coord), abs(r_coord), abs(s_coord)) <= r_max:
                sites.append((q_coord, r_coord))
    return np.asarray(sites, dtype=int)


def axial_to_cartesian(
    axial_sites: np.ndarray,
    *,
    spacing: float,
    alpha_eff: float,
) -> np.ndarray:
    sites = np.asarray(axial_sites, dtype=float).reshape(-1, 2)
    q_coord = sites[:, 0]
    r_coord = sites[:, 1]
    spacing_val = float(spacing)
    alpha_val = float(max(alpha_eff, 1.0e-12))
    x_coord = spacing_val * (q_coord + 0.5 * r_coord)
    y_coord = spacing_val * (np.sqrt(3.0) / 2.0) * r_coord / alpha_val
    return np.column_stack([x_coord, y_coord])


def _dimensionless_trap_scale() -> float:
    l0 = constants.length_harmonic_approximation
    q = constants.ion_charge
    e0 = q**2 / (4.0 * np.pi * constants.epsilon_0 * l0)
    return (q * l0**2) / e0


def find_planar_trap_reference(
    sim: Any,
    plane: PlaneDefinition,
    *,
    initial_guess: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    l0 = constants.length_harmonic_approximation
    q = constants.ion_charge
    e0 = q**2 / (4.0 * np.pi * constants.epsilon_0 * l0)
    lower, upper = get_quartic2d_planar_bounds_arrays()
    bounds = [(float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1]))]
    guess = (
        np.zeros(2, dtype=float)
        if initial_guess is None
        else np.asarray(initial_guess, dtype=float).reshape(2)
    )

    def trap_energy(flat_coords: np.ndarray) -> float:
        point_dimless = plane.coords_to_lab(np.asarray(flat_coords, dtype=float).reshape(1, 2))[0]
        point_si = point_dimless * l0
        return float(q * sim.evaluate_center_poly(*point_si) / e0)

    def trap_gradient(flat_coords: np.ndarray) -> np.ndarray:
        point_dimless = plane.coords_to_lab(np.asarray(flat_coords, dtype=float).reshape(1, 2))[0]
        point_si = point_dimless * l0
        gradient_si = np.asarray(
            sim.evaluate_center_poly_1stderivatives(*point_si),
            dtype=float,
        )
        gradient_dimless = (q * l0 / e0) * gradient_si
        return gradient_dimless @ plane.basis

    result = minimize(
        trap_energy,
        guess,
        jac=trap_gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "gtol": 1.0e-12,
            "ftol": 1.0e-14,
            "maxiter": 4000,
            "maxfun": 40000,
            "maxls": 100,
            "disp": False,
        },
    )
    point_2d = (
        np.asarray(result.x, dtype=float).reshape(2)
        if getattr(result, "x", None) is not None
        else guess
    )
    point_3d_dimless = plane.coords_to_lab(point_2d.reshape(1, 2))[0]
    return point_2d, point_3d_dimless


def build_quartic2d_seed_context(
    sim: Any,
    plane: PlaneDefinition,
) -> PlanarSeedContext:
    reference_point_2d, reference_point_3d_dimless = find_planar_trap_reference(
        sim,
        plane,
    )
    reference_point_si = reference_point_3d_dimless * constants.length_harmonic_approximation
    hessian_si = np.asarray(
        sim.evaluate_center_poly_2ndderivatives(*reference_point_si),
        dtype=float,
    ).reshape(3, 3)
    hessian_dimless = _dimensionless_trap_scale() * hessian_si

    # Quartic2D_101 now keeps the seed aligned with the lab x-z axes.
    # We intentionally ignore x-z cross-coupling here rather than rotate
    # into principal directions.
    curvature_x = float(max(abs(hessian_dimless[0, 0]), 1.0e-12))
    curvature_z = float(max(abs(hessian_dimless[2, 2]), 1.0e-12))
    alpha_eff = float(np.sqrt(curvature_z / curvature_x))

    lower, upper = get_quartic2d_planar_bounds_arrays()

    return PlanarSeedContext(
        reference_point_2d=reference_point_2d,
        reference_point_3d_dimless=reference_point_3d_dimless,
        curvature_x=curvature_x,
        curvature_z=curvature_z,
        alpha_eff=alpha_eff,
        lower=lower,
        upper=upper,
    )


def build_cryo2dcopy_seed_context(
    sim: Any,
    *,
    reference_point_dimless: np.ndarray | None = None,
) -> Cryo2DCopySeedContext:
    point_dimless = (
        np.zeros(3, dtype=float)
        if reference_point_dimless is None
        else np.asarray(reference_point_dimless, dtype=float).reshape(3)
    )
    point_si = point_dimless * constants.length_harmonic_approximation
    hessian_si = np.asarray(
        sim.evaluate_center_poly_2ndderivatives(*point_si),
        dtype=float,
    ).reshape(3, 3)

    # cryo2dcopy intentionally stays in the lab x-z axes.
    curvature_x = float(max(abs(hessian_si[0, 0]), 1.0e-12))
    curvature_z = float(max(abs(hessian_si[2, 2]), 1.0e-12))
    alpha_seed = float(np.sqrt(curvature_z / curvature_x))
    return Cryo2DCopySeedContext(
        reference_point_dimless=point_dimless,
        curvature_x=curvature_x,
        curvature_z=curvature_z,
        alpha_seed=alpha_seed,
    )


def make_cryo_triangular_seed_exact(
    num_ions: int,
    *,
    alpha_yx: float | None = None,
    alpha_eff: float | None = None,
    spacing: float,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> CryoTriangularSeedLocal:
    rng_local = rng or np.random.default_rng()
    alpha_val = _resolve_alpha_yx(alpha_yx=alpha_yx, alpha_eff=alpha_eff)
    radius = minimal_hex_shell_radius(num_ions)
    axial_sites = generate_axial_hex_sites(radius)
    pts = axial_to_cartesian(
        axial_sites,
        spacing=spacing,
        alpha_eff=alpha_val,
    )

    # This helper is intentionally kept semantically identical to the
    # Cryo2D notebook seed helper in local planar coordinates.
    trap_radius = np.sqrt(pts[:, 0] ** 2 + (alpha_val * pts[:, 1]) ** 2)
    keep = np.argsort(trap_radius, kind="stable")[: int(num_ions)]
    pts = pts[keep].copy()

    if jitter > 0.0:
        pts += float(jitter) * float(spacing) * rng_local.standard_normal(pts.shape)

    pts -= pts.mean(axis=0, keepdims=True)
    u0 = pts[:, 0].copy()
    v0 = pts[:, 1].copy()
    uv0 = np.concatenate([u0, v0])
    return CryoTriangularSeedLocal(
        points_2d=pts,
        u0=u0,
        v0=v0,
        uv0=uv0,
    )


def make_cryo_triangular_seed(
    num_ions: int,
    *,
    alpha_yx: float | None = None,
    alpha_eff: float | None = None,
    spacing: float,
    jitter_fraction: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    return make_cryo_triangular_seed_exact(
        num_ions,
        alpha_yx=alpha_yx,
        alpha_eff=alpha_eff,
        spacing=spacing,
        jitter=jitter_fraction,
        rng=rng,
    ).points_2d


def make_cryo2dcopy_triangular_seed(
    num_ions: int,
    alpha_seed: float,
    *,
    spacing: float = 1.0,
    jitter: float = 0.02,
    rng_seed: int = 11,
) -> CryoTriangularSeedLocal:
    rng = np.random.default_rng(int(rng_seed))
    radius = minimal_hex_shell_radius(num_ions)
    axial_sites = generate_axial_hex_sites(radius)
    pts = axial_to_cartesian(
        axial_sites,
        spacing=spacing,
        alpha_eff=alpha_seed,
    )

    trap_radius = np.sqrt(pts[:, 0] ** 2 + (float(alpha_seed) * pts[:, 1]) ** 2)
    keep = np.argsort(trap_radius, kind="stable")[: int(num_ions)]
    pts = pts[keep].copy()

    if jitter > 0.0:
        pts += float(jitter) * float(spacing) * rng.standard_normal(pts.shape)

    pts -= pts.mean(axis=0, keepdims=True)
    u0 = pts[:, 0].copy()
    v0 = pts[:, 1].copy()
    uv0 = np.concatenate([u0, v0])
    return CryoTriangularSeedLocal(
        points_2d=pts,
        u0=u0,
        v0=v0,
        uv0=uv0,
    )


def seed_fits_within_bounds(
    points_2d: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    points = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    lo = np.asarray(lower, dtype=float).reshape(2)
    hi = np.asarray(upper, dtype=float).reshape(2)
    return bool(np.all(points >= lo) and np.all(points <= hi))


def generate_quartic2d_seed_candidates(
    sim: Any,
    num_ions: int,
    *,
    plane: PlaneDefinition | None = None,
    context: PlanarSeedContext | None = None,
    spacing_values: tuple[float, ...] | None = None,
    jitters_per_spacing: int = 3,
    jitter_fraction: float = 2.5e-4,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    plane_def = plane or get_default_quartic2d_plane()
    rng_local = rng or np.random.default_rng()
    seed_context = context or build_quartic2d_seed_context(sim, plane_def)
    spacings = spacing_values or (0.55, 0.75, 1.0, 1.35, 1.8, 2.4, 3.1, 4.0)

    candidates: list[dict[str, Any]] = []
    for spacing in spacings:
        spacing_value = float(spacing)
        for _ in range(int(jitters_per_spacing)):
            local_seed = make_cryo_triangular_seed_exact(
                num_ions,
                alpha_eff=seed_context.alpha_eff,
                spacing=spacing_value,
                jitter=jitter_fraction,
                rng=rng_local,
            )
            # Quartic2D_101 now keeps the seed as a plain Cryo-style blob
            # in lab x-z coordinates. No rotation, mirroring, translation,
            # clipping, or rescaling is applied after construction.
            if not seed_fits_within_bounds(
                local_seed.points_2d,
                seed_context.lower,
                seed_context.upper,
            ):
                continue
            candidates.append(
                {
                    "family": "cryo_triangular",
                    "positions_2d": local_seed.points_2d.copy(),
                    "spacing": spacing_value,
                    "alpha_eff": seed_context.alpha_eff,
                    "seed_fits_bounds_directly": True,
                }
            )
    return candidates
