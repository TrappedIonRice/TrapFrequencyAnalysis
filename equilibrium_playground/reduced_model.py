from __future__ import annotations

"""Reduced even x-z equilibrium model.

The playground uses the dimensionless reduced trap model

    U_trap(x, z) = a20 x^2 + a02 z^2 + a40 x^4 + a22 x^2 z^2 + a04 z^4

with the total multi-ion energy

    U_total = U_trap + sum_{i<j} 1 / r_ij

in purely 2D x-z coordinates. This module contains the direct analytic energy
and gradient evaluation for that model, with no dependency on the main
simulation code.
"""

from dataclasses import dataclass

import numpy as np


_EPS = 1.0e-24


@dataclass(frozen=True)
class ReducedModelCoefficients:
    """Reduced even polynomial coefficients for the 2D x-z trap."""

    a20: float
    a02: float
    a40: float
    a22: float
    a04: float


def split_xz(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a flattened `[x_1..x_N, z_1..z_N]` vector into x/z blocks."""

    vec = np.asarray(q, dtype=float).reshape(-1)
    if vec.size % 2 != 0:
        raise ValueError("q must contain an even number of entries")
    n_ions = vec.size // 2
    return vec[:n_ions], vec[n_ions:]


def pack_xz(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Pack x and z blocks into the flattened optimizer layout."""

    x_arr = np.asarray(x, dtype=float).reshape(-1)
    z_arr = np.asarray(z, dtype=float).reshape(-1)
    if x_arr.shape != z_arr.shape:
        raise ValueError("x and z must have the same shape")
    return np.concatenate([x_arr, z_arr])


def trap_energy(
    q: np.ndarray,
    *,
    a20: float,
    a02: float,
    a40: float,
    a22: float,
    a04: float,
) -> float:
    """Return the reduced even polynomial trap energy."""

    x, z = split_xz(q)
    x2 = x * x
    z2 = z * z
    energy = (
        a20 * np.sum(x2)
        + a02 * np.sum(z2)
        + a40 * np.sum(x2 * x2)
        + a22 * np.sum(x2 * z2)
        + a04 * np.sum(z2 * z2)
    )
    return float(energy)


def trap_gradient(
    q: np.ndarray,
    *,
    a20: float,
    a02: float,
    a40: float,
    a22: float,
    a04: float,
) -> np.ndarray:
    """Return the analytic gradient of the reduced trap energy.

    For one ion at `(x, z)`:

    dU/dx = 2 a20 x + 4 a40 x^3 + 2 a22 x z^2
    dU/dz = 2 a02 z + 2 a22 x^2 z + 4 a04 z^3
    """

    x, z = split_xz(q)
    x2 = x * x
    z2 = z * z
    grad_x = 2.0 * a20 * x + 4.0 * a40 * x2 * x + 2.0 * a22 * x * z2
    grad_z = 2.0 * a02 * z + 2.0 * a22 * x2 * z + 4.0 * a04 * z2 * z
    return pack_xz(grad_x, grad_z)


def coulomb_energy(q: np.ndarray) -> float:
    """Return the 2D Coulomb repulsion energy."""

    x, z = split_xz(q)
    n_ions = x.size
    energy = 0.0
    for i in range(n_ions):
        for j in range(i + 1, n_ions):
            dx = x[i] - x[j]
            dz = z[i] - z[j]
            r2 = dx * dx + dz * dz
            r = np.sqrt(max(r2, _EPS))
            energy += 1.0 / r
    return float(energy)


def coulomb_gradient(q: np.ndarray) -> np.ndarray:
    """Return the analytic gradient of the 2D Coulomb repulsion."""

    x, z = split_xz(q)
    n_ions = x.size
    grad_x = np.zeros(n_ions, dtype=float)
    grad_z = np.zeros(n_ions, dtype=float)
    for i in range(n_ions):
        for j in range(i + 1, n_ions):
            dx = x[i] - x[j]
            dz = z[i] - z[j]
            r2 = max(dx * dx + dz * dz, _EPS)
            r3 = r2 * np.sqrt(r2)
            contrib_x = dx / r3
            contrib_z = dz / r3
            grad_x[i] -= contrib_x
            grad_z[i] -= contrib_z
            grad_x[j] += contrib_x
            grad_z[j] += contrib_z
    return pack_xz(grad_x, grad_z)


def total_energy(
    q: np.ndarray,
    *,
    a20: float,
    a02: float,
    a40: float,
    a22: float,
    a04: float,
) -> float:
    """Return the full reduced-model energy."""

    return trap_energy(q, a20=a20, a02=a02, a40=a40, a22=a22, a04=a04) + coulomb_energy(q)


def total_gradient(
    q: np.ndarray,
    *,
    a20: float,
    a02: float,
    a40: float,
    a22: float,
    a04: float,
) -> np.ndarray:
    """Return the full analytic gradient of the reduced model."""

    return trap_gradient(q, a20=a20, a02=a02, a40=a40, a22=a22, a04=a04) + coulomb_gradient(q)
