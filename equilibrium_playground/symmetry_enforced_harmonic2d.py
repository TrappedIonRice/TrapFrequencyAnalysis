from __future__ import annotations

"""Standalone symmetry-reduced harmonic 2D minimizer for the playground.

The ansatz is a planar x-z crystal with fixed mirror symmetries:

- one ion fixed at the origin
- `num_ions_vert` mirrored pairs on the z-axis
- `num_ions_hor` mirrored pairs on the x-axis
- `num_ions_free` first-quadrant ions mirrored into all four quadrants

This module is intentionally independent from the main simulation stack. It
works directly with manual harmonic trap inputs and evaluates the total energy
and analytic gradient in the symmetry-reduced variables.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import basinhopping, minimize


_EPS = 1.0e-24
_MIN_POSITIVE = 1.0e-9


@dataclass(frozen=True)
class SymmetryAnsatzCounts:
    """Symmetry-orbit counts for the reduced planar ansatz."""

    num_ions_vert: int
    num_ions_hor: int
    num_ions_free: int

    def __post_init__(self) -> None:
        for name, value in (
            ("num_ions_vert", self.num_ions_vert),
            ("num_ions_hor", self.num_ions_hor),
            ("num_ions_free", self.num_ions_free),
        ):
            if int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    @property
    def total_ion_count(self) -> int:
        """Return `1 + 2*vert + 2*hor + 4*free`."""

        return (
            1
            + 2 * int(self.num_ions_vert)
            + 2 * int(self.num_ions_hor)
            + 4 * int(self.num_ions_free)
        )

    @property
    def n_params(self) -> int:
        """Return `vert + hor + 2*free`."""

        return int(self.num_ions_vert) + int(self.num_ions_hor) + 2 * int(self.num_ions_free)


@dataclass(frozen=True)
class HarmonicTrap2D:
    """Normalized harmonic trap parameters used by the symmetry solver.

    The trap is

        U_trap = sum_i (a_x * x_i^2 + a_z * z_i^2)

    and the frequency interpretation in this standalone playground is

        a_x = omega_x^2
        a_z = omega_z^2

    with `omega_x = scale` and `omega_z = alpha * scale` in the ratio-plus-scale
    input mode.
    """

    input_mode: str
    omega_x: float
    omega_z: float
    alpha: float
    scale: float
    a_x: float
    a_z: float

    def to_metadata(self) -> dict[str, float | str]:
        return {
            "trap_input_mode": self.input_mode,
            "omega_x": float(self.omega_x),
            "omega_z": float(self.omega_z),
            "alpha": float(self.alpha),
            "scale": float(self.scale),
            "a_x": float(self.a_x),
            "a_z": float(self.a_z),
        }


@dataclass(frozen=True)
class SymmetryReducedState:
    """Physical symmetry variables after positivity/ordering reconstruction."""

    vert_distances: np.ndarray
    hor_distances: np.ndarray
    free_x: np.ndarray
    free_z: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        free_points = (
            np.column_stack((self.free_x, self.free_z))
            if self.free_x.size
            else np.zeros((0, 2), dtype=float)
        )
        return {
            "vert_distances": self.vert_distances.copy(),
            "hor_distances": self.hor_distances.copy(),
            "free_points_q1": free_points,
        }


@dataclass(frozen=True)
class DecodedSymmetryVariables:
    """Decoded optimizer variables and chain-rule data."""

    state: SymmetryReducedState
    vert_increment_derivatives: np.ndarray
    hor_increment_derivatives: np.ndarray
    free_x_derivatives: np.ndarray
    free_z_derivatives: np.ndarray


@dataclass(frozen=True)
class SymmetryEnforcedSeed:
    """Single seed for one symmetry-enforced optimization run."""

    raw_variables: np.ndarray
    reduced_state: SymmetryReducedState
    positions_2d: np.ndarray
    seed_family: str
    base_spacing: float
    alpha_yx: float
    shell_radius: int
    seed_scale: float
    vertical_seed_scale_factor: float
    seed_jitter: float
    rng_seed: int | None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "seed_family": self.seed_family,
            "base_spacing": float(self.base_spacing),
            "alpha_yx": float(self.alpha_yx),
            "shell_radius": int(self.shell_radius),
            "seed_scale": float(self.seed_scale),
            "vertical_seed_scale_factor": float(self.vertical_seed_scale_factor),
            "seed_jitter": float(self.seed_jitter),
            "rng_seed": self.rng_seed,
        }


def _build_seed_from_raw_variables(
    counts: SymmetryAnsatzCounts,
    raw_variables: np.ndarray,
    *,
    seed_metadata: dict[str, Any] | None = None,
) -> SymmetryEnforcedSeed:
    """Build a seed object from externally supplied raw optimizer variables."""

    metadata = {} if seed_metadata is None else dict(seed_metadata)
    raw = np.asarray(raw_variables, dtype=float).reshape(-1)
    decoded = decode_raw_symmetry_variables(raw, counts)
    state = decoded.state
    rng_seed = metadata.get("rng_seed", None)
    return SymmetryEnforcedSeed(
        raw_variables=raw.copy(),
        reduced_state=state,
        positions_2d=expand_symmetric_coordinates(state),
        seed_family=str(metadata.get("seed_family", "external_raw_variables")),
        base_spacing=float(metadata.get("base_spacing", np.nan)),
        alpha_yx=float(metadata.get("alpha_yx", np.nan)),
        shell_radius=int(metadata.get("shell_radius", -1)),
        seed_scale=float(metadata.get("seed_scale", np.nan)),
        vertical_seed_scale_factor=float(
            metadata.get("vertical_seed_scale_factor", np.nan)
        ),
        seed_jitter=float(metadata.get("seed_jitter", np.nan)),
        rng_seed=None if rng_seed is None else int(rng_seed),
    )


@dataclass(frozen=True)
class RawObjectiveEvaluation:
    """Cached energy/gradient evaluation in both raw and physical variables."""

    raw_variables: np.ndarray
    decoded: DecodedSymmetryVariables
    physical_gradient: np.ndarray
    raw_gradient: np.ndarray
    energy: float


@dataclass
class SymmetryEnforcedHarmonic2DResult:
    """Result bundle for the standalone symmetry-enforced harmonic solver."""

    positions_2d: np.ndarray
    energy: float
    success: bool
    message: str
    optimizer_path: str
    trap_input_mode: str
    trap_parameters: dict[str, float | str]
    num_ions_vert: int
    num_ions_hor: int
    num_ions_free: int
    total_ion_count: int
    reduced_variables: dict[str, np.ndarray]
    optimizer_variables_raw: np.ndarray
    seed_positions_2d: np.ndarray
    seed_reduced_variables: dict[str, np.ndarray]
    seed_optimizer_variables_raw: np.ndarray
    seed_metadata: dict[str, Any]
    reduced_gradient_max_norm: float
    raw_gradient_max_norm: float
    seed_to_final_rms_move: float
    nfev: int | None = None
    njev: int | None = None
    nit: int | None = None
    basin_iterations: int | None = None
    basinhopping_success: bool | None = None
    basinhopping_message: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "minimizer_name": "SymmetryEnforced_Harmonic2D",
            "optimizer_name": self.optimizer_path,
            "optimizer_path": self.optimizer_path,
            "trap_input_mode": self.trap_input_mode,
            "energy": float(self.energy),
            "success": bool(self.success),
            "message": str(self.message),
            "num_ions_vert": int(self.num_ions_vert),
            "num_ions_hor": int(self.num_ions_hor),
            "num_ions_free": int(self.num_ions_free),
            "total_ion_count": int(self.total_ion_count),
            "reduced_gradient_max_norm": float(self.reduced_gradient_max_norm),
            "raw_gradient_max_norm": float(self.raw_gradient_max_norm),
            "distance_moved_from_seed_dimless": float(self.seed_to_final_rms_move),
            "seed_metadata": dict(self.seed_metadata),
            "trap_parameters": dict(self.trap_parameters),
            "reduced_variables": {
                key: np.asarray(value, dtype=float).copy()
                for key, value in self.reduced_variables.items()
            },
            "positions_2d": self.positions_2d.copy(),
        }
        return metadata


def symmetry_total_ion_count(
    num_ions_vert: int,
    num_ions_hor: int,
    num_ions_free: int,
) -> int:
    """Return the expanded ion count for the symmetry ansatz."""

    return SymmetryAnsatzCounts(num_ions_vert, num_ions_hor, num_ions_free).total_ion_count


def normalize_harmonic_trap(
    *,
    omega_x: float | None = None,
    omega_z: float | None = None,
    alpha: float | None = None,
    scale: float | None = None,
) -> HarmonicTrap2D:
    """Normalize the supported trap input modes into one harmonic representation.

    Supported modes:

    1. Direct frequencies:
       `omega_x`, `omega_z`

    2. Ratio plus scale:
       `alpha`, `scale`

       interpreted as

           omega_x = scale
           omega_z = alpha * scale

    In this standalone playground the harmonic coefficients are

        a_x = omega_x^2
        a_z = omega_z^2
    """

    direct_supplied = omega_x is not None or omega_z is not None
    ratio_supplied = alpha is not None or scale is not None
    if direct_supplied and ratio_supplied:
        raise ValueError("provide either (omega_x, omega_z) or (alpha, scale), not both")
    if not direct_supplied and not ratio_supplied:
        raise ValueError("trap parameters are required")

    if direct_supplied:
        if omega_x is None or omega_z is None:
            raise ValueError("both omega_x and omega_z must be provided together")
        omega_x_value = float(omega_x)
        omega_z_value = float(omega_z)
        if omega_x_value <= 0.0 or omega_z_value <= 0.0:
            raise ValueError("omega_x and omega_z must be positive")
        input_mode = "direct_omega"
        alpha_value = omega_z_value / omega_x_value
        scale_value = omega_x_value
    else:
        if alpha is None or scale is None:
            raise ValueError("both alpha and scale must be provided together")
        alpha_value = float(alpha)
        scale_value = float(scale)
        if alpha_value <= 0.0 or scale_value <= 0.0:
            raise ValueError("alpha and scale must be positive")
        input_mode = "alpha_scale"
        omega_x_value = scale_value
        omega_z_value = alpha_value * scale_value

    return HarmonicTrap2D(
        input_mode=input_mode,
        omega_x=omega_x_value,
        omega_z=omega_z_value,
        alpha=alpha_value,
        scale=scale_value,
        a_x=omega_x_value * omega_x_value,
        a_z=omega_z_value * omega_z_value,
    )


def _softplus(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(arr))) + np.maximum(arr, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    out = np.empty_like(arr)
    positive = arr >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_x = np.exp(arr[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _positive_from_raw(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = _softplus(np.asarray(raw, dtype=float)) + _MIN_POSITIVE
    derivatives = _sigmoid(np.asarray(raw, dtype=float))
    return values, derivatives


def _raw_from_positive(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - _MIN_POSITIVE
    if np.any(shifted <= 0.0):
        raise ValueError("positive values must exceed the minimum positivity floor")
    raw = np.empty_like(shifted)
    large = shifted > 20.0
    raw[large] = shifted[large]
    raw[~large] = np.log(np.expm1(shifted[~large]))
    return raw


def pack_physical_symmetry_variables(state: SymmetryReducedState) -> np.ndarray:
    """Pack the physical reduced variables as `[vert, hor, free_x, free_z]`."""

    return np.concatenate(
        [
            np.asarray(state.vert_distances, dtype=float),
            np.asarray(state.hor_distances, dtype=float),
            np.asarray(state.free_x, dtype=float),
            np.asarray(state.free_z, dtype=float),
        ]
    )


def unpack_physical_symmetry_variables(
    vector: np.ndarray,
    counts: SymmetryAnsatzCounts,
) -> SymmetryReducedState:
    """Unpack the physical reduced-variable vector."""

    flat = np.asarray(vector, dtype=float).reshape(-1)
    if flat.size != counts.n_params:
        raise ValueError("physical variable vector has the wrong size")

    cursor = 0
    num_vert = int(counts.num_ions_vert)
    num_hor = int(counts.num_ions_hor)
    num_free = int(counts.num_ions_free)

    vert = flat[cursor : cursor + num_vert].copy()
    cursor += num_vert
    hor = flat[cursor : cursor + num_hor].copy()
    cursor += num_hor
    free_x = flat[cursor : cursor + num_free].copy()
    cursor += num_free
    free_z = flat[cursor : cursor + num_free].copy()

    return SymmetryReducedState(
        vert_distances=vert,
        hor_distances=hor,
        free_x=free_x,
        free_z=free_z,
    )


def decode_raw_symmetry_variables(
    raw_variables: np.ndarray,
    counts: SymmetryAnsatzCounts,
) -> DecodedSymmetryVariables:
    """Decode unconstrained optimizer variables into physical symmetry variables.

    Axis distances use positive increments plus cumulative sums, which guarantees
    strict positivity and strict outward ordering. First-quadrant free-ion
    coordinates use a simple positivity transform with no additional ordering.
    """

    raw = np.asarray(raw_variables, dtype=float).reshape(-1)
    if raw.size != counts.n_params:
        raise ValueError("raw variable vector has the wrong size")

    cursor = 0
    num_vert = int(counts.num_ions_vert)
    num_hor = int(counts.num_ions_hor)
    num_free = int(counts.num_ions_free)

    raw_vert = raw[cursor : cursor + num_vert]
    cursor += num_vert
    raw_hor = raw[cursor : cursor + num_hor]
    cursor += num_hor
    raw_free_x = raw[cursor : cursor + num_free]
    cursor += num_free
    raw_free_z = raw[cursor : cursor + num_free]

    vert_increments, vert_increment_derivatives = _positive_from_raw(raw_vert)
    hor_increments, hor_increment_derivatives = _positive_from_raw(raw_hor)
    free_x, free_x_derivatives = _positive_from_raw(raw_free_x)
    free_z, free_z_derivatives = _positive_from_raw(raw_free_z)

    return DecodedSymmetryVariables(
        state=SymmetryReducedState(
            vert_distances=np.cumsum(vert_increments),
            hor_distances=np.cumsum(hor_increments),
            free_x=free_x,
            free_z=free_z,
        ),
        vert_increment_derivatives=vert_increment_derivatives,
        hor_increment_derivatives=hor_increment_derivatives,
        free_x_derivatives=free_x_derivatives,
        free_z_derivatives=free_z_derivatives,
    )


def encode_seed_state_to_raw(state: SymmetryReducedState) -> np.ndarray:
    """Encode a positive physical seed state back into unconstrained variables."""

    vert = np.asarray(state.vert_distances, dtype=float)
    hor = np.asarray(state.hor_distances, dtype=float)
    free_x = np.asarray(state.free_x, dtype=float)
    free_z = np.asarray(state.free_z, dtype=float)

    vert_increments = np.diff(np.concatenate(([0.0], vert))) if vert.size else np.zeros(0)
    hor_increments = np.diff(np.concatenate(([0.0], hor))) if hor.size else np.zeros(0)
    return np.concatenate(
        [
            _raw_from_positive(vert_increments),
            _raw_from_positive(hor_increments),
            _raw_from_positive(free_x),
            _raw_from_positive(free_z),
        ]
    )


def expand_symmetric_coordinates(state: SymmetryReducedState) -> np.ndarray:
    """Expand the reduced symmetry variables into full x-z coordinates."""

    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for distance in np.asarray(state.vert_distances, dtype=float):
        points.append((0.0, float(distance)))
        points.append((0.0, -float(distance)))
    for distance in np.asarray(state.hor_distances, dtype=float):
        points.append((float(distance), 0.0))
        points.append((-float(distance), 0.0))
    for x_coord, z_coord in zip(
        np.asarray(state.free_x, dtype=float),
        np.asarray(state.free_z, dtype=float),
        strict=True,
    ):
        x_value = float(x_coord)
        z_value = float(z_coord)
        points.extend(
            [
                (x_value, z_value),
                (-x_value, z_value),
                (x_value, -z_value),
                (-x_value, -z_value),
            ]
        )
    return np.asarray(points, dtype=float)


def _safe_distance(value: float) -> float:
    return float(max(value, _EPS))


def _safe_radius(dx: float, dz: float) -> float:
    return float(np.sqrt(max(dx * dx + dz * dz, _EPS)))


def _gradient_max_norm(gradient: np.ndarray) -> float:
    grad = np.asarray(gradient, dtype=float).reshape(-1)
    if grad.size == 0:
        return 0.0
    return float(np.max(np.abs(grad)))


def _seed_to_final_rms_move(
    final_positions_2d: np.ndarray,
    seed_positions_2d: np.ndarray,
) -> float:
    final_points = np.asarray(final_positions_2d, dtype=float).reshape(-1, 2)
    seed_points = np.asarray(seed_positions_2d, dtype=float).reshape(-1, 2)
    diff = final_points - seed_points
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def symmetry_trap_energy(state: SymmetryReducedState, trap: HarmonicTrap2D) -> float:
    """Return the direct harmonic trap energy in reduced variables."""

    vert = np.asarray(state.vert_distances, dtype=float)
    hor = np.asarray(state.hor_distances, dtype=float)
    free_x = np.asarray(state.free_x, dtype=float)
    free_z = np.asarray(state.free_z, dtype=float)
    energy = (
        2.0 * trap.a_z * np.sum(vert * vert)
        + 2.0 * trap.a_x * np.sum(hor * hor)
        + 4.0 * trap.a_x * np.sum(free_x * free_x)
        + 4.0 * trap.a_z * np.sum(free_z * free_z)
    )
    return float(energy)


def symmetry_trap_gradient(state: SymmetryReducedState, trap: HarmonicTrap2D) -> np.ndarray:
    """Return the direct harmonic trap gradient in physical reduced variables."""

    vert = np.asarray(state.vert_distances, dtype=float)
    hor = np.asarray(state.hor_distances, dtype=float)
    free_x = np.asarray(state.free_x, dtype=float)
    free_z = np.asarray(state.free_z, dtype=float)
    return np.concatenate(
        [
            4.0 * trap.a_z * vert,
            4.0 * trap.a_x * hor,
            8.0 * trap.a_x * free_x,
            8.0 * trap.a_z * free_z,
        ]
    )


def symmetry_coulomb_energy(state: SymmetryReducedState) -> float:
    """Return the symmetry-reduced Coulomb repulsion energy."""

    vert = np.asarray(state.vert_distances, dtype=float)
    hor = np.asarray(state.hor_distances, dtype=float)
    free_x = np.asarray(state.free_x, dtype=float)
    free_z = np.asarray(state.free_z, dtype=float)

    energy = 0.0

    for distance in vert:
        v = _safe_distance(float(distance))
        energy += 2.0 / v
        energy += 1.0 / (2.0 * v)

    for distance in hor:
        h = _safe_distance(float(distance))
        energy += 2.0 / h
        energy += 1.0 / (2.0 * h)

    for i in range(vert.size):
        vi = float(vert[i])
        for j in range(i + 1, vert.size):
            vj = float(vert[j])
            diff = _safe_distance(vj - vi)
            summation = _safe_distance(vi + vj)
            energy += 2.0 / diff + 2.0 / summation

    for i in range(hor.size):
        hi = float(hor[i])
        for j in range(i + 1, hor.size):
            hj = float(hor[j])
            diff = _safe_distance(hj - hi)
            summation = _safe_distance(hi + hj)
            energy += 2.0 / diff + 2.0 / summation

    for v in vert:
        for h in hor:
            radius = _safe_radius(float(h), float(v))
            energy += 4.0 / radius

    for idx in range(free_x.size):
        x_value = float(free_x[idx])
        z_value = float(free_z[idx])
        origin_radius = _safe_radius(x_value, z_value)
        energy += 4.0 / origin_radius
        energy += 1.0 / _safe_distance(x_value)
        energy += 1.0 / _safe_distance(z_value)
        energy += 1.0 / origin_radius

    for v in vert:
        v_value = float(v)
        for idx in range(free_x.size):
            x_value = float(free_x[idx])
            z_value = float(free_z[idx])
            r_minus = _safe_radius(x_value, v_value - z_value)
            r_plus = _safe_radius(x_value, v_value + z_value)
            energy += 4.0 / r_minus + 4.0 / r_plus

    for h in hor:
        h_value = float(h)
        for idx in range(free_x.size):
            x_value = float(free_x[idx])
            z_value = float(free_z[idx])
            r_minus = _safe_radius(h_value - x_value, z_value)
            r_plus = _safe_radius(h_value + x_value, z_value)
            energy += 4.0 / r_minus + 4.0 / r_plus

    for i in range(free_x.size):
        x_i = float(free_x[i])
        z_i = float(free_z[i])
        for j in range(i + 1, free_x.size):
            x_j = float(free_x[j])
            z_j = float(free_z[j])
            r_mm = _safe_radius(x_i - x_j, z_i - z_j)
            r_mp = _safe_radius(x_i - x_j, z_i + z_j)
            r_pm = _safe_radius(x_i + x_j, z_i - z_j)
            r_pp = _safe_radius(x_i + x_j, z_i + z_j)
            energy += 4.0 * (1.0 / r_mm + 1.0 / r_mp + 1.0 / r_pm + 1.0 / r_pp)

    return float(energy)


def symmetry_coulomb_gradient(state: SymmetryReducedState) -> np.ndarray:
    """Return the analytic Coulomb gradient in physical reduced variables."""

    vert = np.asarray(state.vert_distances, dtype=float)
    hor = np.asarray(state.hor_distances, dtype=float)
    free_x = np.asarray(state.free_x, dtype=float)
    free_z = np.asarray(state.free_z, dtype=float)

    grad_vert = np.zeros_like(vert)
    grad_hor = np.zeros_like(hor)
    grad_free_x = np.zeros_like(free_x)
    grad_free_z = np.zeros_like(free_z)

    for idx, distance in enumerate(vert):
        v = _safe_distance(float(distance))
        grad_vert[idx] += -2.0 / (v * v)
        grad_vert[idx] += -1.0 / (2.0 * v * v)

    for idx, distance in enumerate(hor):
        h = _safe_distance(float(distance))
        grad_hor[idx] += -2.0 / (h * h)
        grad_hor[idx] += -1.0 / (2.0 * h * h)

    for i in range(vert.size):
        vi = float(vert[i])
        for j in range(i + 1, vert.size):
            vj = float(vert[j])
            diff = _safe_distance(vj - vi)
            diff_sq = diff * diff
            summation = _safe_distance(vi + vj)
            sum_sq = summation * summation
            grad_vert[i] += 2.0 / diff_sq - 2.0 / sum_sq
            grad_vert[j] += -2.0 / diff_sq - 2.0 / sum_sq

    for i in range(hor.size):
        hi = float(hor[i])
        for j in range(i + 1, hor.size):
            hj = float(hor[j])
            diff = _safe_distance(hj - hi)
            diff_sq = diff * diff
            summation = _safe_distance(hi + hj)
            sum_sq = summation * summation
            grad_hor[i] += 2.0 / diff_sq - 2.0 / sum_sq
            grad_hor[j] += -2.0 / diff_sq - 2.0 / sum_sq

    for i, v in enumerate(vert):
        v_value = float(v)
        for j, h in enumerate(hor):
            h_value = float(h)
            radius = _safe_radius(h_value, v_value)
            inv_r3 = 1.0 / (radius * radius * radius)
            grad_vert[i] += -4.0 * v_value * inv_r3
            grad_hor[j] += -4.0 * h_value * inv_r3

    for idx in range(free_x.size):
        x_value = float(free_x[idx])
        z_value = float(free_z[idx])
        origin_radius = _safe_radius(x_value, z_value)
        inv_r3 = 1.0 / (origin_radius * origin_radius * origin_radius)
        grad_free_x[idx] += -4.0 * x_value * inv_r3
        grad_free_z[idx] += -4.0 * z_value * inv_r3
        grad_free_x[idx] += -1.0 / (_safe_distance(x_value) ** 2)
        grad_free_z[idx] += -1.0 / (_safe_distance(z_value) ** 2)
        grad_free_x[idx] += -x_value * inv_r3
        grad_free_z[idx] += -z_value * inv_r3

    for i, v in enumerate(vert):
        v_value = float(v)
        for j in range(free_x.size):
            x_value = float(free_x[j])
            z_value = float(free_z[j])
            r_minus = _safe_radius(x_value, v_value - z_value)
            r_plus = _safe_radius(x_value, v_value + z_value)
            inv_minus_r3 = 1.0 / (r_minus * r_minus * r_minus)
            inv_plus_r3 = 1.0 / (r_plus * r_plus * r_plus)
            grad_vert[i] += -4.0 * (v_value - z_value) * inv_minus_r3
            grad_vert[i] += -4.0 * (v_value + z_value) * inv_plus_r3
            grad_free_x[j] += -4.0 * x_value * inv_minus_r3
            grad_free_x[j] += -4.0 * x_value * inv_plus_r3
            grad_free_z[j] += 4.0 * (v_value - z_value) * inv_minus_r3
            grad_free_z[j] += -4.0 * (v_value + z_value) * inv_plus_r3

    for i, h in enumerate(hor):
        h_value = float(h)
        for j in range(free_x.size):
            x_value = float(free_x[j])
            z_value = float(free_z[j])
            delta = h_value - x_value
            sigma = h_value + x_value
            r_minus = _safe_radius(delta, z_value)
            r_plus = _safe_radius(sigma, z_value)
            inv_minus_r3 = 1.0 / (r_minus * r_minus * r_minus)
            inv_plus_r3 = 1.0 / (r_plus * r_plus * r_plus)
            grad_hor[i] += -4.0 * delta * inv_minus_r3
            grad_hor[i] += -4.0 * sigma * inv_plus_r3
            grad_free_x[j] += 4.0 * delta * inv_minus_r3
            grad_free_x[j] += -4.0 * sigma * inv_plus_r3
            grad_free_z[j] += -4.0 * z_value * inv_minus_r3
            grad_free_z[j] += -4.0 * z_value * inv_plus_r3

    for i in range(free_x.size):
        x_i = float(free_x[i])
        z_i = float(free_z[i])
        for j in range(i + 1, free_x.size):
            x_j = float(free_x[j])
            z_j = float(free_z[j])
            dx_minus = x_i - x_j
            dx_plus = x_i + x_j
            dz_minus = z_i - z_j
            dz_plus = z_i + z_j

            r_mm = _safe_radius(dx_minus, dz_minus)
            r_mp = _safe_radius(dx_minus, dz_plus)
            r_pm = _safe_radius(dx_plus, dz_minus)
            r_pp = _safe_radius(dx_plus, dz_plus)

            inv_mm_r3 = 4.0 / (r_mm * r_mm * r_mm)
            inv_mp_r3 = 4.0 / (r_mp * r_mp * r_mp)
            inv_pm_r3 = 4.0 / (r_pm * r_pm * r_pm)
            inv_pp_r3 = 4.0 / (r_pp * r_pp * r_pp)

            grad_free_x[i] += -dx_minus * inv_mm_r3
            grad_free_x[i] += -dx_minus * inv_mp_r3
            grad_free_x[i] += -dx_plus * inv_pm_r3
            grad_free_x[i] += -dx_plus * inv_pp_r3

            grad_free_x[j] += dx_minus * inv_mm_r3
            grad_free_x[j] += dx_minus * inv_mp_r3
            grad_free_x[j] += -dx_plus * inv_pm_r3
            grad_free_x[j] += -dx_plus * inv_pp_r3

            grad_free_z[i] += -dz_minus * inv_mm_r3
            grad_free_z[i] += -dz_plus * inv_mp_r3
            grad_free_z[i] += -dz_minus * inv_pm_r3
            grad_free_z[i] += -dz_plus * inv_pp_r3

            grad_free_z[j] += dz_minus * inv_mm_r3
            grad_free_z[j] += -dz_plus * inv_mp_r3
            grad_free_z[j] += dz_minus * inv_pm_r3
            grad_free_z[j] += -dz_plus * inv_pp_r3

    return np.concatenate([grad_vert, grad_hor, grad_free_x, grad_free_z])


def symmetry_total_energy(state: SymmetryReducedState, trap: HarmonicTrap2D) -> float:
    """Return the full direct energy in the symmetry-reduced variables."""

    return symmetry_trap_energy(state, trap) + symmetry_coulomb_energy(state)


def symmetry_total_gradient(state: SymmetryReducedState, trap: HarmonicTrap2D) -> np.ndarray:
    """Return the full direct gradient in physical reduced variables."""

    return symmetry_trap_gradient(state, trap) + symmetry_coulomb_gradient(state)


def expanded_total_energy(points_2d: np.ndarray, trap: HarmonicTrap2D) -> float:
    """Return the brute-force full energy of an expanded x-z configuration.

    This helper is mainly for validation and lightweight testing.
    """

    points = np.asarray(points_2d, dtype=float).reshape(-1, 2)
    x = points[:, 0]
    z = points[:, 1]
    energy = float(np.sum(trap.a_x * x * x + trap.a_z * z * z))
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            radius = _safe_radius(x[i] - x[j], z[i] - z[j])
            energy += 1.0 / radius
    return float(energy)


def _physical_gradient_to_raw_gradient(
    physical_gradient: np.ndarray,
    decoded: DecodedSymmetryVariables,
    counts: SymmetryAnsatzCounts,
) -> np.ndarray:
    grad = np.asarray(physical_gradient, dtype=float).reshape(-1)
    unpacked = unpack_physical_symmetry_variables(grad, counts)

    grad_vert_increments = np.cumsum(unpacked.vert_distances[::-1])[::-1]
    grad_hor_increments = np.cumsum(unpacked.hor_distances[::-1])[::-1]

    return np.concatenate(
        [
            decoded.vert_increment_derivatives * grad_vert_increments,
            decoded.hor_increment_derivatives * grad_hor_increments,
            decoded.free_x_derivatives * unpacked.free_x,
            decoded.free_z_derivatives * unpacked.free_z,
        ]
    )


class _RawSymmetryObjective:
    """Cached objective in raw optimizer variables."""

    def __init__(self, counts: SymmetryAnsatzCounts, trap: HarmonicTrap2D) -> None:
        self.counts = counts
        self.trap = trap
        self._cached: RawObjectiveEvaluation | None = None

    def evaluate(self, raw_variables: np.ndarray) -> RawObjectiveEvaluation:
        raw = np.asarray(raw_variables, dtype=float).reshape(-1)
        cached = self._cached
        if cached is not None and np.array_equal(raw, cached.raw_variables):
            return cached

        decoded = decode_raw_symmetry_variables(raw, self.counts)
        physical_gradient = symmetry_total_gradient(decoded.state, self.trap)
        raw_gradient = _physical_gradient_to_raw_gradient(
            physical_gradient,
            decoded,
            self.counts,
        )
        evaluation = RawObjectiveEvaluation(
            raw_variables=raw.copy(),
            decoded=decoded,
            physical_gradient=physical_gradient,
            raw_gradient=raw_gradient,
            energy=symmetry_total_energy(decoded.state, self.trap),
        )
        self._cached = evaluation
        return evaluation

    def energy(self, raw_variables: np.ndarray) -> float:
        return float(self.evaluate(raw_variables).energy)

    def gradient(self, raw_variables: np.ndarray) -> np.ndarray:
        return self.evaluate(raw_variables).raw_gradient.copy()


def _generate_axial_hex_sites(radius: int) -> np.ndarray:
    r_max = int(radius)
    if r_max < 0:
        raise ValueError("radius must be nonnegative")

    sites = []
    for q_coord in range(-r_max, r_max + 1):
        for r_coord in range(-r_max, r_max + 1):
            s_coord = -q_coord - r_coord
            if max(abs(q_coord), abs(r_coord), abs(s_coord)) <= r_max:
                sites.append((q_coord, r_coord))
    return np.asarray(sites, dtype=int)


def _extract_lattice_orbit_candidates(
    radius: int,
    *,
    base_spacing: float,
    alpha_yx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizontal_candidates: list[tuple[float, float]] = []
    vertical_candidates: list[tuple[float, float]] = []
    free_candidates: list[tuple[float, float, float]] = []

    alpha = float(max(alpha_yx, _EPS))
    spacing = float(base_spacing)
    z_prefactor = spacing * (np.sqrt(3.0) / 2.0) / alpha

    for q_coord, r_coord in _generate_axial_hex_sites(radius):
        x_coord = spacing * (q_coord + 0.5 * r_coord)
        z_coord = z_prefactor * r_coord
        trap_radius = float(np.sqrt(x_coord * x_coord + (alpha * z_coord) * (alpha * z_coord)))

        if r_coord == 0 and q_coord > 0:
            horizontal_candidates.append((trap_radius, float(x_coord)))
        elif (2 * q_coord + r_coord) == 0 and r_coord > 0:
            vertical_candidates.append((trap_radius, float(z_coord)))
        elif r_coord > 0 and (2 * q_coord + r_coord) > 0:
            free_candidates.append((trap_radius, float(x_coord), float(z_coord)))

    horizontal_candidates.sort(key=lambda item: (item[0], item[1]))
    vertical_candidates.sort(key=lambda item: (item[0], item[1]))
    free_candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    horizontal = (
        np.asarray([item[1] for item in horizontal_candidates], dtype=float)
        if horizontal_candidates
        else np.zeros(0, dtype=float)
    )
    vertical = (
        np.asarray([item[1] for item in vertical_candidates], dtype=float)
        if vertical_candidates
        else np.zeros(0, dtype=float)
    )
    free = (
        np.asarray([[item[1], item[2]] for item in free_candidates], dtype=float)
        if free_candidates
        else np.zeros((0, 2), dtype=float)
    )
    return horizontal, vertical, free


def _select_vertical_axis_seed_distances(
    vertical_candidates: np.ndarray,
    count: int,
    *,
    vertical_seed_scale_factor: float,
) -> np.ndarray | None:
    num_vertical = int(count)
    if num_vertical <= 0:
        return np.zeros(0, dtype=float)

    factor = float(vertical_seed_scale_factor)
    if factor < 1.0:
        raise ValueError("vertical_seed_scale_factor must be at least 1.0")

    target_indices = (
        np.ceil(factor * np.arange(1, num_vertical + 1, dtype=float)).astype(int) - 1
    )
    if vertical_candidates.size == 0 or int(target_indices[-1]) >= int(vertical_candidates.size):
        return None
    return vertical_candidates[target_indices].copy()


def _build_coherent_lattice_seed_state(
    counts: SymmetryAnsatzCounts,
    *,
    base_spacing: float,
    alpha_yx: float,
    vertical_seed_scale_factor: float,
) -> tuple[SymmetryReducedState, int]:
    radius = max(
        1,
        int(counts.num_ions_hor),
        int(np.ceil(2.0 * float(vertical_seed_scale_factor) * int(counts.num_ions_vert))),
    )
    while True:
        horizontal_candidates, vertical_candidates, free_candidates = _extract_lattice_orbit_candidates(
            radius,
            base_spacing=base_spacing,
            alpha_yx=alpha_yx,
        )
        selected_vertical = _select_vertical_axis_seed_distances(
            vertical_candidates,
            int(counts.num_ions_vert),
            vertical_seed_scale_factor=vertical_seed_scale_factor,
        )
        if (
            horizontal_candidates.size >= int(counts.num_ions_hor)
            and selected_vertical is not None
            and free_candidates.shape[0] >= int(counts.num_ions_free)
        ):
            state = SymmetryReducedState(
                vert_distances=selected_vertical,
                hor_distances=horizontal_candidates[: int(counts.num_ions_hor)].copy(),
                free_x=free_candidates[: int(counts.num_ions_free), 0].copy(),
                free_z=free_candidates[: int(counts.num_ions_free), 1].copy(),
            )
            return state, radius
        radius += 1


def make_symmetry_enforced_seed(
    counts: SymmetryAnsatzCounts,
    trap: HarmonicTrap2D,
    *,
    seed_scale: float = 1.0,
    vertical_seed_scale_factor: float = 1.0,
    seed_jitter: float = 0.0,
    rng_seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> SymmetryEnforcedSeed:
    """Build a coherent symmetry seed from one anisotropic triangular lattice.

    The seed geometry follows the Cryo-style anisotropic triangular lattice:

        x = a * (q + r/2)
        z = a * (sqrt(3)/2) * r / alpha_yx

    with one common base spacing `a`. The horizontal-axis pairs, vertical-axis
    pairs, and first-quadrant quartet representatives are all selected from
    that same lattice. `vertical_seed_scale_factor` keeps this coherent lattice
    geometry but chooses the vertical-axis orbit representatives farther out on
    the same lattice. A value of `1.0` preserves the current baseline seed.
    """

    if seed_scale <= 0.0:
        raise ValueError("seed_scale must be positive")
    if vertical_seed_scale_factor < 1.0:
        raise ValueError("vertical_seed_scale_factor must be at least 1.0")
    if seed_jitter < 0.0:
        raise ValueError("seed_jitter must be nonnegative")

    alpha_yx = float(max(trap.alpha, _EPS))
    base_spacing = float(seed_scale) * np.power(max(trap.a_x, _EPS), -1.0 / 3.0)
    state, shell_radius = _build_coherent_lattice_seed_state(
        counts,
        base_spacing=base_spacing,
        alpha_yx=alpha_yx,
        vertical_seed_scale_factor=vertical_seed_scale_factor,
    )
    vert = state.vert_distances.copy()
    hor = state.hor_distances.copy()
    free_x = state.free_x.copy()
    free_z = state.free_z.copy()

    if seed_jitter > 0.0:
        rng_obj = np.random.default_rng(rng_seed) if rng is None else rng
        if vert.size:
            increments = np.diff(np.concatenate(([0.0], vert)))
            increments *= np.exp(float(seed_jitter) * rng_obj.standard_normal(increments.shape))
            vert = np.cumsum(increments)
        if hor.size:
            increments = np.diff(np.concatenate(([0.0], hor)))
            increments *= np.exp(float(seed_jitter) * rng_obj.standard_normal(increments.shape))
            hor = np.cumsum(increments)
        if free_x.size:
            free_x *= np.exp(float(seed_jitter) * rng_obj.standard_normal(free_x.shape))
            free_z *= np.exp(float(seed_jitter) * rng_obj.standard_normal(free_z.shape))

    state = SymmetryReducedState(
        vert_distances=vert,
        hor_distances=hor,
        free_x=free_x,
        free_z=free_z,
    )

    if np.any(state.vert_distances <= 0.0) or np.any(np.diff(state.vert_distances) <= 0.0):
        raise RuntimeError("coherent lattice seed produced invalid vertical-axis distances")
    if np.any(state.hor_distances <= 0.0) or np.any(np.diff(state.hor_distances) <= 0.0):
        raise RuntimeError("coherent lattice seed produced invalid horizontal-axis distances")
    if np.any(state.free_x <= 0.0) or np.any(state.free_z <= 0.0):
        raise RuntimeError("coherent lattice seed produced invalid first-quadrant points")

    return SymmetryEnforcedSeed(
        raw_variables=encode_seed_state_to_raw(state),
        reduced_state=state,
        positions_2d=expand_symmetric_coordinates(state),
        seed_family="symmetry_coherent_lattice_orbits",
        base_spacing=base_spacing,
        alpha_yx=alpha_yx,
        shell_radius=int(shell_radius),
        seed_scale=float(seed_scale),
        vertical_seed_scale_factor=float(vertical_seed_scale_factor),
        seed_jitter=float(seed_jitter),
        rng_seed=rng_seed,
    )


def solve_symmetry_enforced_harmonic2d(
    num_ions_vert: int,
    num_ions_hor: int,
    num_ions_free: int,
    *,
    omega_x: float | None = None,
    omega_z: float | None = None,
    alpha: float | None = None,
    scale: float | None = None,
    seed_scale: float = 1.0,
    vertical_seed_scale_factor: float = 1.0,
    seed_jitter: float = 0.0,
    rng_seed: int | None = None,
    initial_raw_variables: np.ndarray | None = None,
    initial_seed_metadata: dict[str, Any] | None = None,
    basinhopping_niter: int = 60,
    basin_temperature: float = 0.35,
    basin_stepsize: float = 0.3,
    bfgs_gtol: float = 1.0e-10,
    bfgs_maxiter: int = 1000,
    final_gtol: float = 1.0e-11,
    final_ftol: float = 1.0e-14,
    final_maxiter: int = 4000,
) -> SymmetryEnforcedHarmonic2DResult:
    """Run the standalone symmetry-reduced harmonic 2D minimizer.

    The solver uses one seed, one basin-hopping stage, and one final local
    refinement stage. The code is intentionally structured so repeated-run
    wrappers can be added later without changing the solver core.

    If `initial_raw_variables` is provided, the solver starts from that
    externally supplied raw optimizer state instead of constructing the generic
    symmetry seed. This is used by the scan workflow for alpha-direction
    warm-starting.
    """

    counts = SymmetryAnsatzCounts(num_ions_vert, num_ions_hor, num_ions_free)
    trap = normalize_harmonic_trap(
        omega_x=omega_x,
        omega_z=omega_z,
        alpha=alpha,
        scale=scale,
    )

    if initial_raw_variables is None:
        seed = make_symmetry_enforced_seed(
            counts,
            trap,
            seed_scale=seed_scale,
            vertical_seed_scale_factor=vertical_seed_scale_factor,
            seed_jitter=seed_jitter,
            rng_seed=rng_seed,
        )
    else:
        seed = _build_seed_from_raw_variables(
            counts,
            initial_raw_variables,
            seed_metadata=initial_seed_metadata,
        )
    objective = _RawSymmetryObjective(counts, trap)

    bh_kwargs = {
        "method": "BFGS",
        "jac": objective.gradient,
        "options": {
            "gtol": float(bfgs_gtol),
            "disp": False,
            "maxiter": int(bfgs_maxiter),
        },
    }
    bh_res = basinhopping(
        objective.energy,
        seed.raw_variables,
        minimizer_kwargs=bh_kwargs,
        niter=int(basinhopping_niter),
        T=float(basin_temperature),
        stepsize=float(basin_stepsize),
        disp=False,
        seed=rng_seed,
    )

    refine_res = minimize(
        objective.energy,
        np.asarray(bh_res.x, dtype=float),
        jac=objective.gradient,
        method="L-BFGS-B",
        options={
            "gtol": float(final_gtol),
            "ftol": float(final_ftol),
            "maxiter": int(final_maxiter),
            "maxls": 50,
        },
    )

    final_raw = (
        np.asarray(refine_res.x, dtype=float)
        if getattr(refine_res, "x", None) is not None
        else np.asarray(bh_res.x, dtype=float)
    )
    final_eval = objective.evaluate(final_raw)
    final_state = final_eval.decoded.state
    final_positions = expand_symmetric_coordinates(final_state)

    bh_lowest = getattr(bh_res, "lowest_optimization_result", None)
    bh_success = bool(getattr(bh_lowest, "success", False)) if bh_lowest is not None else None
    bh_message = str(getattr(bh_lowest, "message", getattr(bh_res, "message", "")))

    return SymmetryEnforcedHarmonic2DResult(
        positions_2d=final_positions,
        energy=float(getattr(refine_res, "fun", final_eval.energy)),
        success=bool(getattr(refine_res, "success", False)),
        message=str(getattr(refine_res, "message", "")),
        optimizer_path="basinhopping(BFGS)->L-BFGS-B",
        trap_input_mode=trap.input_mode,
        trap_parameters=trap.to_metadata(),
        num_ions_vert=int(counts.num_ions_vert),
        num_ions_hor=int(counts.num_ions_hor),
        num_ions_free=int(counts.num_ions_free),
        total_ion_count=int(counts.total_ion_count),
        reduced_variables=final_state.to_dict(),
        optimizer_variables_raw=final_raw.copy(),
        seed_positions_2d=seed.positions_2d.copy(),
        seed_reduced_variables=seed.reduced_state.to_dict(),
        seed_optimizer_variables_raw=seed.raw_variables.copy(),
        seed_metadata=seed.to_metadata(),
        reduced_gradient_max_norm=_gradient_max_norm(final_eval.physical_gradient),
        raw_gradient_max_norm=_gradient_max_norm(final_eval.raw_gradient),
        seed_to_final_rms_move=_seed_to_final_rms_move(final_positions, seed.positions_2d),
        nfev=getattr(refine_res, "nfev", None),
        njev=getattr(refine_res, "njev", None),
        nit=getattr(refine_res, "nit", None),
        basin_iterations=int(basinhopping_niter),
        basinhopping_success=bh_success,
        basinhopping_message=bh_message,
    )
