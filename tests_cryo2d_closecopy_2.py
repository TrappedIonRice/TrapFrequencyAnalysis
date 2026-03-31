from types import SimpleNamespace

import numpy as np

import constants
from sim.cryo2d_closecopy_2 import (
    ReducedEvenTrapModel,
    build_reduced_even_trap_model,
    find_1ion_planar_minimum,
    grad_U2D,
    make_triangular_seed,
    reduced_even_trap_energy,
    reduced_even_trap_gradient,
)


def _energy_scales() -> tuple[float, float, float]:
    l0 = constants.length_harmonic_approximation
    q = constants.ion_charge
    e0 = q**2 / (4.0 * np.pi * constants.epsilon_0 * l0)
    return l0, q, e0


class HarmonicTrapOnlySim:
    def __init__(self, *, kx=0.55, ky=2.5, kz=1.6):
        self.kx = float(kx)
        self.ky = float(ky)
        self.kz = float(kz)
        self.trapVariables = SimpleNamespace(dc_key="dc")
        self.center_fits = {}

    def _ensure_dc_center_fit(self, polyfit_deg=4):
        return None

    def evaluate_center_poly(self, x, y, z):
        return 0.5 * (self.kx * x**2 + self.ky * y**2 + self.kz * z**2)

    def evaluate_center_poly_1stderivatives(self, x, y, z):
        return (self.kx * x, self.ky * y, self.kz * z)

    def evaluate_center_poly_2ndderivatives(self, x, y, z):
        return np.array(
            [
                [self.kx, 0.0, 0.0],
                [0.0, self.ky, 0.0],
                [0.0, 0.0, self.kz],
            ],
            dtype=float,
        )

    def evaluate_center_poly_4thderivatives(self, x, y, z):
        return np.zeros((3, 3, 3, 3), dtype=float)


class ReducedPolynomialStubSim:
    def __init__(self):
        self.trapVariables = SimpleNamespace(dc_key="dc")
        self.center_fits = {}
        self.c10 = 1.7
        self.c01 = -2.3
        self.c20 = 0.6
        self.c11 = 4.1
        self.c02 = 0.9
        self.c30 = -1.4
        self.c21 = 2.6
        self.c12 = -3.2
        self.c03 = 5.5
        self.c40 = 0.11
        self.c22 = 0.17
        self.c04 = 0.23

    def _ensure_dc_center_fit(self, polyfit_deg=4):
        return None

    def evaluate_center_poly(self, x, y, z):
        return (
            self.c10 * x
            + self.c01 * z
            + self.c20 * x**2
            + self.c11 * x * z
            + self.c02 * z**2
            + self.c30 * x**3
            + self.c21 * x**2 * z
            + self.c12 * x * z**2
            + self.c03 * z**3
            + self.c40 * x**4
            + self.c22 * x**2 * z**2
            + self.c04 * z**4
        )

    def evaluate_center_poly_1stderivatives(self, x, y, z):
        dx = (
            self.c10
            + 2.0 * self.c20 * x
            + self.c11 * z
            + 3.0 * self.c30 * x**2
            + 2.0 * self.c21 * x * z
            + self.c12 * z**2
            + 4.0 * self.c40 * x**3
            + 2.0 * self.c22 * x * z**2
        )
        dz = (
            self.c01
            + self.c11 * x
            + 2.0 * self.c02 * z
            + self.c21 * x**2
            + 2.0 * self.c12 * x * z
            + 3.0 * self.c03 * z**2
            + 2.0 * self.c22 * x**2 * z
            + 4.0 * self.c04 * z**3
        )
        return (dx, 0.0, dz)

    def evaluate_center_poly_2ndderivatives(self, x, y, z):
        dxx = (
            2.0 * self.c20
            + 6.0 * self.c30 * x
            + 2.0 * self.c21 * z
            + 12.0 * self.c40 * x**2
            + 2.0 * self.c22 * z**2
        )
        dxz = (
            self.c11
            + 2.0 * self.c21 * x
            + 2.0 * self.c12 * z
            + 4.0 * self.c22 * x * z
        )
        dzz = (
            2.0 * self.c02
            + 2.0 * self.c12 * x
            + 6.0 * self.c03 * z
            + 2.0 * self.c22 * x**2
            + 12.0 * self.c04 * z**2
        )
        return np.array(
            [
                [dxx, 0.0, dxz],
                [0.0, 0.0, 0.0],
                [dxz, 0.0, dzz],
            ],
            dtype=float,
        )

    def evaluate_center_poly_4thderivatives(self, x, y, z):
        tensor = np.zeros((3, 3, 3, 3), dtype=float)
        tensor[0, 0, 0, 0] = 24.0 * self.c40
        tensor[2, 2, 2, 2] = 24.0 * self.c04
        tensor[0, 0, 2, 2] = 4.0 * self.c22
        tensor[0, 2, 0, 2] = 4.0 * self.c22
        tensor[0, 2, 2, 0] = 4.0 * self.c22
        tensor[2, 0, 0, 2] = 4.0 * self.c22
        tensor[2, 0, 2, 0] = 4.0 * self.c22
        tensor[2, 2, 0, 0] = 4.0 * self.c22
        return tensor


def test_build_reduced_even_trap_model_keeps_only_even_terms():
    sim = ReducedPolynomialStubSim()
    model = build_reduced_even_trap_model(sim, np.zeros(2, dtype=float))
    l0, q, e0 = _energy_scales()
    scale2 = q * l0**2 / e0
    scale4 = q * l0**4 / e0

    assert np.isclose(model.a20, scale2 * sim.c20)
    assert np.isclose(model.a02, scale2 * sim.c02)
    assert np.isclose(model.a40, scale4 * sim.c40)
    assert np.isclose(model.a22, scale4 * sim.c22)
    assert np.isclose(model.a04, scale4 * sim.c04)


def test_reduced_even_trap_gradient_matches_finite_difference():
    model = ReducedEvenTrapModel(
        center_xz_dimless=np.zeros(2, dtype=float),
        center_3d_dimless=np.zeros(3, dtype=float),
        curvature_x=1.0,
        curvature_z=2.0,
        alpha_seed=np.sqrt(2.0),
        a20=0.8,
        a02=1.1,
        a40=0.05,
        a22=0.07,
        a04=0.09,
    )
    uv = np.array([0.2, -0.35, 0.15, -0.1], dtype=float)
    analytic = reduced_even_trap_gradient(uv, model)

    eps = 1.0e-7
    numeric = np.zeros_like(uv)
    for idx in range(len(uv)):
        step = np.zeros_like(uv)
        step[idx] = eps
        numeric[idx] = (
            reduced_even_trap_energy(uv + step, model)
            - reduced_even_trap_energy(uv - step, model)
        ) / (2.0 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1.0e-6, atol=1.0e-8)


def test_find_1ion_planar_minimum_is_origin_for_harmonic_trap():
    sim = HarmonicTrapOnlySim(kx=0.5, ky=3.0, kz=2.0)
    center = find_1ion_planar_minimum(sim)
    np.testing.assert_allclose(center, np.zeros(2), atol=1.0e-12)


def test_make_triangular_seed_centers_com_and_defaults():
    seed = make_triangular_seed(9, 1.8, spacing=1.0, jitter=0.02, rng_seed=11)
    assert seed.spacing == 1.0
    assert seed.jitter == 0.02
    assert seed.rng_seed == 11
    np.testing.assert_allclose(seed.points_2d.mean(axis=0), np.zeros(2), atol=1.0e-14)


def test_grad_u2d_projects_out_com_mode():
    model = ReducedEvenTrapModel(
        center_xz_dimless=np.zeros(2, dtype=float),
        center_3d_dimless=np.zeros(3, dtype=float),
        curvature_x=1.0,
        curvature_z=4.0,
        alpha_seed=2.0,
        a20=0.8,
        a02=1.3,
        a40=0.02,
        a22=0.05,
        a04=0.03,
    )
    uv = np.array([0.4, -0.2, 0.1, 0.25, -0.15, 0.05], dtype=float)
    grad = grad_U2D(uv, model)
    assert grad.shape == uv.shape
    assert np.isclose(np.sum(grad[:3]), 0.0)
    assert np.isclose(np.sum(grad[3:]), 0.0)
