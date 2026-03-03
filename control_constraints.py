"""
Utilities to build linear equality constraints on polynomial coefficients.

Example
    import numpy as np
    from control_constraints import build_target_hessian, build_L_b_for_point

    # Given: polynomial powers (M x 3) and voltage-to-c mapping A (M x K)
    r0 = np.array([0.0, 0.0, 0.0])
    freqs = np.array([2*np.pi*1e6, 2*np.pi*1.2e6, 2*np.pi*0.8e6])  # rad/s
    R = np.eye(3)
    Kstar = build_target_hessian(freqs, R, mass=m, charge=q, poly_is_potential_energy=False)
    L, b = build_L_b_for_point(powers, r0, Kstar)

    # Solve for control voltages u given c = A @ u and constraints L @ c = b
    # => (L @ A) @ u = b
    u, *_ = np.linalg.lstsq(L @ A, b, rcond=None)
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import constants


def build_target_hessian(
    freqs_hz_or_rad: Sequence[float],
    principal_dirs: Sequence[Sequence[float]] | np.ndarray,
    mass: float,
    charge: float | None = None,
    poly_is_potential_energy: bool = True,
    *,
    freqs_in_hz: bool = False,
) -> np.ndarray:
    """
    Build target Hessian at equilibrium from mode frequencies and principal axes.

    If freqs_in_hz is True, convert to rad/s via omega = 2*pi*f.
    If poly_is_potential_energy is False, charge is required and the Hessian
    corresponds to electric potential (V) rather than energy (J).
    """
    freqs = np.asarray(freqs_hz_or_rad, dtype=float).reshape(-1)
    if freqs.shape[0] != 3:
        raise ValueError("freqs_hz_or_rad must have length 3")
    if freqs_in_hz:
        freqs = 2.0 * np.pi * freqs

    R = _as_rotation_matrix(principal_dirs)
    omega2 = freqs ** 2
    diag = np.diag(omega2)

    if poly_is_potential_energy:
        scale = float(mass)
    else:
        if charge is None:
            raise ValueError("charge is required when poly_is_potential_energy is False")
        scale = float(mass) / float(charge)

    Kstar = scale * (R @ diag @ R.T)
    return Kstar


def build_L_b_for_point(
    powers: np.ndarray,
    r0: Sequence[float],
    Kstar: np.ndarray,
    include_gradient: bool = True,
    hess_order: str = "vech",
    basis: str = "nondim",
    nd_L0_m: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build linear constraints L @ c = b enforcing gradient=0 and Hessian=Kstar.

    hess_order="vech" uses rows [xx, yy, zz, xy, xz, yz].
    If basis="nondim", r0 is scaled by L0 and Kstar is scaled by L0^2.
    """
    p = np.asarray(powers, dtype=int)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("powers must be (M, 3)")
    r0v = np.asarray(r0, dtype=float).reshape(-1)
    if r0v.shape[0] != 3:
        raise ValueError("r0 must have length 3")
    K = np.asarray(Kstar, dtype=float)
    if K.shape != (3, 3):
        raise ValueError("Kstar must be 3x3")
    if hess_order != "vech":
        raise ValueError("only hess_order='vech' is supported")

    if basis not in ("nondim", "physical"):
        raise ValueError("basis must be 'nondim' or 'physical'")
    L0 = float(constants.ND_L0_M if nd_L0_m is None else nd_L0_m)
    if basis == "nondim":
        r_eval = r0v / L0
        K_use = K * (L0**2)
    else:
        r_eval = r0v
        K_use = K

    dx, dy, dz = build_derivative_rows(p, r_eval)
    dxx, dyy, dzz, dxy, dxz, dyz = build_hessian_rows(p, r_eval)

    hess_rows = [dxx, dyy, dzz, dxy, dxz, dyz]
    if include_gradient:
        L = np.vstack([dx, dy, dz] + hess_rows).astype(float)
        b = np.zeros(9, dtype=float)
        b[3:] = _vech_symmetric(K_use)
    else:
        L = np.vstack(hess_rows).astype(float)
        b = _vech_symmetric(K_use)

    return L, b


def _as_rotation_matrix(principal_dirs: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix with orthonormal columns."""
    arr = np.asarray(principal_dirs, dtype=float)
    if arr.shape == (3, 3):
        mat = arr
    else:
        if len(arr) != 3:
            raise ValueError("principal_dirs must be 3x3 or a list of three vectors")
        mat = np.column_stack(arr)
        if mat.shape != (3, 3):
            raise ValueError("principal_dirs vectors must be length 3")

    # Orthonormalize for safety
    q, _ = np.linalg.qr(mat)
    if np.linalg.det(q) < 0:
        q[:, 2] *= -1.0
    return q


def _vech_symmetric(M3: np.ndarray) -> np.ndarray:
    """Vectorize symmetric 3x3 as [xx, yy, zz, xy, xz, yz]."""
    return np.array(
        [M3[0, 0], M3[1, 1], M3[2, 2], M3[0, 1], M3[0, 2], M3[1, 2]],
        dtype=float,
    )


def _monomial_eval(p: Sequence[int], r0: Sequence[float]) -> float:
    px, py, pz = (int(p[0]), int(p[1]), int(p[2]))
    x0, y0, z0 = (float(r0[0]), float(r0[1]), float(r0[2]))
    return (x0 ** px) * (y0 ** py) * (z0 ** pz)


def build_derivative_rows(powers: np.ndarray, r0: Sequence[float]) -> Tuple[np.ndarray, ...]:
    """Return dx, dy, dz rows for all monomials at r0."""
    p = np.asarray(powers, dtype=int)
    r0v = np.asarray(r0, dtype=float).reshape(-1)
    x0, y0, z0 = (float(r0v[0]), float(r0v[1]), float(r0v[2]))
    M = p.shape[0]

    dx = np.zeros(M, dtype=float)
    dy = np.zeros(M, dtype=float)
    dz = np.zeros(M, dtype=float)
    for j in range(M):
        px, py, pz = (int(p[j, 0]), int(p[j, 1]), int(p[j, 2]))
        if px >= 1:
            dx[j] = px * (x0 ** (px - 1)) * (y0 ** py) * (z0 ** pz)
        if py >= 1:
            dy[j] = py * (x0 ** px) * (y0 ** (py - 1)) * (z0 ** pz)
        if pz >= 1:
            dz[j] = pz * (x0 ** px) * (y0 ** py) * (z0 ** (pz - 1))
    return dx, dy, dz


def build_hessian_rows(powers: np.ndarray, r0: Sequence[float]) -> Tuple[np.ndarray, ...]:
    """Return dxx, dyy, dzz, dxy, dxz, dyz rows for all monomials at r0."""
    p = np.asarray(powers, dtype=int)
    r0v = np.asarray(r0, dtype=float).reshape(-1)
    x0, y0, z0 = (float(r0v[0]), float(r0v[1]), float(r0v[2]))
    M = p.shape[0]

    dxx = np.zeros(M, dtype=float)
    dyy = np.zeros(M, dtype=float)
    dzz = np.zeros(M, dtype=float)
    dxy = np.zeros(M, dtype=float)
    dxz = np.zeros(M, dtype=float)
    dyz = np.zeros(M, dtype=float)

    for j in range(M):
        px, py, pz = (int(p[j, 0]), int(p[j, 1]), int(p[j, 2]))
        if px >= 2:
            dxx[j] = px * (px - 1) * (x0 ** (px - 2)) * (y0 ** py) * (z0 ** pz)
        if py >= 2:
            dyy[j] = py * (py - 1) * (x0 ** px) * (y0 ** (py - 2)) * (z0 ** pz)
        if pz >= 2:
            dzz[j] = pz * (pz - 1) * (x0 ** px) * (y0 ** py) * (z0 ** (pz - 2))
        if px >= 1 and py >= 1:
            dxy[j] = px * py * (x0 ** (px - 1)) * (y0 ** (py - 1)) * (z0 ** pz)
        if px >= 1 and pz >= 1:
            dxz[j] = px * pz * (x0 ** (px - 1)) * (y0 ** py) * (z0 ** (pz - 1))
        if py >= 1 and pz >= 1:
            dyz[j] = py * pz * (x0 ** px) * (y0 ** (py - 1)) * (z0 ** (pz - 1))

    return dxx, dyy, dzz, dxy, dxz, dyz


if __name__ == "__main__":
    # Hard-coded sanity run: build Kstar from freqs, principal directions, r0.
    powers = np.array(
        [
            [2, 0, 0],  # x^2
            [0, 2, 0],  # y^2
            [0, 0, 2],  # z^2
            [1, 1, 0],  # xy
            [1, 0, 1],  # xz
            [0, 1, 1],  # yz
            [1, 0, 0],  # x
            [0, 1, 0],  # y
            [0, 0, 1],  # z
            [0, 0, 0],  # constant
        ],
        dtype=int,
    )

    r0 = np.array([0.0, 0.0, 0.0], dtype=float)
    freqs = np.array([2 * np.pi * 1e6, 2 * np.pi * 1.2e6, 2 * np.pi * 0.8e6])
    principal_dirs = [[.5,.1,.7],[1,0,0],[.3,.3,.3]]
    mass = 6.64e-26  # example: 40Ca+ in kg
    charge = 1.602176634e-19  # elementary charge in C

    Kstar = build_target_hessian(
        freqs,
        principal_dirs,
        mass=mass,
        charge=charge,
        poly_is_potential_energy=False,
    )

    L, b = build_L_b_for_point(powers, r0, Kstar, include_gradient=True)
    np.set_printoptions(precision=6, suppress=True)
    print("L shape:", L.shape)
    print("L:")
    print(L)
    print("b shape:", b.shape)
    print("b:")
    print(b)
