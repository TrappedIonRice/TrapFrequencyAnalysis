from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, basinhopping, minimize

import constants


CRYO2D_CLOSECOPY_2 = "Cryo2d_closecopy_2"
_BOUNDS_SCALE = 3.0
_EPS = 1.0e-12
_PLANE_NORMAL = np.array([0.0, 1.0, 0.0], dtype=float)
_OPTIMIZER_PATH = "basinhopping(BFGS)->trust-constr"
_SEED_SPACING = 1.0
_SEED_JITTER = 0.02
_SEED_RNG_SEED = 11
_TRAP_MODEL_NAME = "reduced_even_x2_z2_x4_x2z2_z4"


@dataclass(frozen=True)
class ReducedEvenTrapModel:
    center_xz_dimless: np.ndarray
    center_3d_dimless: np.ndarray
    curvature_x: float
    curvature_z: float
    alpha_seed: float
    a20: float
    a02: float
    a40: float
    a22: float
    a04: float


@dataclass(frozen=True)
class Cryo2DTriangularSeed:
    points_2d: np.ndarray
    u0: np.ndarray
    v0: np.ndarray
    uv0: np.ndarray
    spacing: float
    jitter: float
    rng_seed: int


@dataclass
class Cryo2DCloseCopy2Result:
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
    trap_model: str
    reduced_even_model: bool
    a20: float
    a02: float
    a40: float
    a22: float
    a04: float
    nfev: int | None = None
    njev: int | None = None
    nit: int | None = None
    basin_iterations: int | None = None
    num_seed_attempts: int = 1
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
            "projected_gradient_max_norm": float(
                self.projected_gradient_max_norm
            ),
            "distance_moved_from_seed_dimless": float(
                self.distance_moved_from_seed_dimless
            ),
            "trap_model": str(self.trap_model),
            "reduced_even_model": bool(self.reduced_even_model),
            "a20": float(self.a20),
            "a02": float(self.a02),
            "a40": float(self.a40),
            "a22": float(self.a22),
            "a04": float(self.a04),
            "nfev": self.nfev,
            "njev": self.njev,
            "nit": self.nit,
            "basin_iterations": self.basin_iterations,
            "num_seed_attempts": int(self.num_seed_attempts),
            "minimum_pair_separation": self.minimum_pair_separation,
        }


def _energy_scales() -> tuple[float, float, float]:
    l0 = constants.length_harmonic_approximation
    q = constants.ion_charge
    e0 = q**2 / (4.0 * np.pi * constants.epsilon_0 * l0)
    return l0, q, e0


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
    return np.clip(
        np.asarray(uv, dtype=float).reshape(-1),
        np.asarray(lower_local, dtype=float).reshape(-1),
        np.asarray(upper_local, dtype=float).reshape(-1),
    )


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


def _projected_gradient_max_norm(uv_grad: np.ndarray) -> float:
    grad = np.asarray(uv_grad, dtype=float).reshape(-1)
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


def _trap_only_energy_dimless(
    xz_dimless: np.ndarray,
    sim: Any,
) -> float:
    l0, q, e0 = _energy_scales()
    xz = np.asarray(xz_dimless, dtype=float).reshape(2)
    point_si = np.array([xz[0], 0.0, xz[1]], dtype=float) * l0
    return float(q * sim.evaluate_center_poly(*point_si) / e0)


def _trap_only_gradient_dimless(
    xz_dimless: np.ndarray,
    sim: Any,
) -> np.ndarray:
    l0, q, e0 = _energy_scales()
    xz = np.asarray(xz_dimless, dtype=float).reshape(2)
    point_si = np.array([xz[0], 0.0, xz[1]], dtype=float) * l0
    d1 = np.asarray(sim.evaluate_center_poly_1stderivatives(*point_si), dtype=float)
    scale1 = q * l0 / e0
    return scale1 * np.array([d1[0], d1[2]], dtype=float)


def find_1ion_planar_minimum(sim: Any) -> np.ndarray:
    if hasattr(sim, "_ensure_dc_center_fit"):
        sim._ensure_dc_center_fit(polyfit_deg=4)
    lower, upper = _expanded_bounds_xz()
    result = minimize(
        lambda xz: _trap_only_energy_dimless(xz, sim),
        np.zeros(2, dtype=float),
        jac=lambda xz: _trap_only_gradient_dimless(xz, sim),
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
    if getattr(result, "x", None) is None:
        return np.zeros(2, dtype=float)
    return np.asarray(result.x, dtype=float).reshape(2)


def build_reduced_even_trap_model(
    sim: Any,
    center_xz_dimless: np.ndarray,
) -> ReducedEvenTrapModel:
    l0, q, e0 = _energy_scales()
    center_xz = np.asarray(center_xz_dimless, dtype=float).reshape(2)
    center_3d_dimless = np.array([center_xz[0], 0.0, center_xz[1]], dtype=float)
    point_si = center_3d_dimless * l0

    h2_si = np.asarray(
        sim.evaluate_center_poly_2ndderivatives(*point_si),
        dtype=float,
    ).reshape(3, 3)
    h4_si = np.asarray(
        sim.evaluate_center_poly_4thderivatives(*point_si),
        dtype=float,
    ).reshape(3, 3, 3, 3)

    curvature_x = float(max(abs(h2_si[0, 0]), _EPS))
    curvature_z = float(max(abs(h2_si[2, 2]), _EPS))
    alpha_seed = float(np.sqrt(curvature_z / curvature_x))

    scale2 = q * l0**2 / e0
    scale4 = q * l0**4 / e0

    return ReducedEvenTrapModel(
        center_xz_dimless=center_xz,
        center_3d_dimless=center_3d_dimless,
        curvature_x=curvature_x,
        curvature_z=curvature_z,
        alpha_seed=alpha_seed,
        a20=0.5 * scale2 * float(h2_si[0, 0]),
        a02=0.5 * scale2 * float(h2_si[2, 2]),
        a40=(1.0 / 24.0) * scale4 * float(h4_si[0, 0, 0, 0]),
        a22=0.25 * scale4 * float(h4_si[0, 0, 2, 2]),
        a04=(1.0 / 24.0) * scale4 * float(h4_si[2, 2, 2, 2]),
    )


def reduced_even_trap_energy(
    uv: np.ndarray,
    model: ReducedEvenTrapModel,
) -> float:
    vec = np.asarray(uv, dtype=float).reshape(-1)
    num_ions = len(vec) // 2
    u = vec[:num_ions]
    v = vec[num_ions:]
    return float(
        np.sum(
            model.a20 * u**2
            + model.a02 * v**2
            + model.a40 * u**4
            + model.a22 * u**2 * v**2
            + model.a04 * v**4
        )
    )


def reduced_even_trap_gradient(
    uv: np.ndarray,
    model: ReducedEvenTrapModel,
) -> np.ndarray:
    vec = np.asarray(uv, dtype=float).reshape(-1)
    num_ions = len(vec) // 2
    u = vec[:num_ions]
    v = vec[num_ions:]
    du = (
        2.0 * model.a20 * u
        + 4.0 * model.a40 * u**3
        + 2.0 * model.a22 * u * v**2
    )
    dv = (
        2.0 * model.a02 * v
        + 2.0 * model.a22 * u**2 * v
        + 4.0 * model.a04 * v**3
    )
    return np.concatenate([du, dv])


def _coulomb_energy(
    uv: np.ndarray,
) -> float:
    vec = np.asarray(uv, dtype=float).reshape(-1)
    num_ions = len(vec) // 2
    u = vec[:num_ions]
    v = vec[num_ions:]
    energy = 0.0
    for i in range(num_ions):
        for j in range(i + 1, num_ions):
            dx = u[i] - u[j]
            dz = v[i] - v[j]
            r = np.hypot(dx, dz)
            if r < _EPS:
                r = 1.0e-8
            energy += 1.0 / r
    return float(energy)


def _coulomb_gradient(
    uv: np.ndarray,
) -> np.ndarray:
    vec = np.asarray(uv, dtype=float).reshape(-1)
    num_ions = len(vec) // 2
    u = vec[:num_ions]
    v = vec[num_ions:]
    gu = np.zeros(num_ions, dtype=float)
    gv = np.zeros(num_ions, dtype=float)
    for i in range(num_ions):
        for j in range(num_ions):
            if i == j:
                continue
            dx = u[i] - u[j]
            dz = v[i] - v[j]
            r2 = dx * dx + dz * dz
            r2 = max(r2, 1.0e-24)
            r3 = r2 * np.sqrt(r2)
            gu[i] -= dx / r3
            gv[i] -= dz / r3
    return np.concatenate([gu, gv])


def U2D(
    uv: np.ndarray,
    model: ReducedEvenTrapModel,
) -> float:
    return reduced_even_trap_energy(uv, model) + _coulomb_energy(uv)


def grad_U2D(
    uv: np.ndarray,
    model: ReducedEvenTrapModel,
) -> np.ndarray:
    grad = reduced_even_trap_gradient(uv, model) + _coulomb_gradient(uv)
    num_ions = len(grad) // 2
    gu = grad[:num_ions].copy()
    gv = grad[num_ions:].copy()
    gu -= np.mean(gu)
    gv -= np.mean(gv)
    return np.concatenate([gu, gv])


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
            float(spacing)
            * (np.sqrt(3.0) / 2.0)
            * r_coord
            / float(max(alpha_yx, _EPS))
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


def solve_cryo2d_closecopy_2(
    sim: Any,
    num_ions: int,
) -> Cryo2DCloseCopy2Result:
    if hasattr(sim, "_ensure_dc_center_fit"):
        sim._ensure_dc_center_fit(polyfit_deg=4)

    center_xz = find_1ion_planar_minimum(sim)
    model = build_reduced_even_trap_model(sim, center_xz)
    seed = make_triangular_seed(
        int(num_ions),
        model.alpha_seed,
        spacing=_SEED_SPACING,
        jitter=_SEED_JITTER,
        rng_seed=_SEED_RNG_SEED,
    )

    energy_fn = lambda x: U2D(x, model)
    gradient_fn = lambda x: grad_U2D(x, model)
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
        seed=_SEED_RNG_SEED,
    )

    lower_local, upper_local = _build_local_bounds_vectors(
        int(num_ions),
        model.center_xz_dimless,
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

    final_positions_dimless = _local_uv_to_lab_positions(final_uv, model.center_xz_dimless)
    final_positions_dimless, order = _sort_positions_xz(final_positions_dimless)
    final_uv_sorted = _positions_to_local_uv(final_positions_dimless, model.center_xz_dimless)

    seed_positions_dimless = _local_uv_to_lab_positions(seed.uv0, model.center_xz_dimless)
    seed_positions_dimless, seed_order = _sort_positions_xz(seed_positions_dimless)
    seed_uv_sorted = _positions_to_local_uv(seed_positions_dimless, model.center_xz_dimless)

    final_positions_si = final_positions_dimless * constants.length_harmonic_approximation
    seed_positions_si = seed_positions_dimless * constants.length_harmonic_approximation
    grad_norm = _projected_gradient_max_norm(gradient_fn(final_uv_sorted))
    success = bool(getattr(refine_res, "success", False)) or bool(grad_norm <= 1.0e-8)

    return Cryo2DCloseCopy2Result(
        positions=final_positions_si,
        positions_dimless=final_positions_dimless,
        positions_local_dimless=np.column_stack(
            [final_uv_sorted[: int(num_ions)], final_uv_sorted[int(num_ions) :]]
        ),
        energy=float(getattr(refine_res, "fun", energy_fn(final_uv_sorted))),
        success=success,
        message=str(getattr(refine_res, "message", "")),
        optimizer_name=_OPTIMIZER_PATH,
        minimizer_name=CRYO2D_CLOSECOPY_2,
        plane_normal=_PLANE_NORMAL.copy(),
        seed_family="cryo_triangular",
        seed_positions=seed_positions_si,
        seed_positions_dimless=seed_positions_dimless,
        seed_positions_local_dimless=np.column_stack(
            [seed_uv_sorted[: int(num_ions)], seed_uv_sorted[int(num_ions) :]]
        ),
        seed_spacing=_SEED_SPACING,
        seed_jitter=_SEED_JITTER,
        seed_rng_seed=_SEED_RNG_SEED,
        alpha_eff=float(model.alpha_seed),
        reference_point_dimless=np.asarray(model.center_3d_dimless, dtype=float),
        stage="final_polish",
        optimizer_path=_OPTIMIZER_PATH,
        projected_gradient_max_norm=grad_norm,
        distance_moved_from_seed_dimless=_distance_moved_from_seed_dimless(
            final_uv_sorted,
            seed_uv_sorted,
        ),
        trap_model=_TRAP_MODEL_NAME,
        reduced_even_model=True,
        a20=float(model.a20),
        a02=float(model.a02),
        a40=float(model.a40),
        a22=float(model.a22),
        a04=float(model.a04),
        nfev=getattr(refine_res, "nfev", None),
        njev=getattr(refine_res, "njev", None),
        nit=getattr(refine_res, "nit", getattr(refine_res, "niter", None)),
        basin_iterations=getattr(bh_res, "nit", None),
        num_seed_attempts=1,
        minimum_pair_separation=_minimum_pair_separation(final_positions_si),
    )
