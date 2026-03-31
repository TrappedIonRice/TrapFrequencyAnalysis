from types import SimpleNamespace

import numpy as np

from sim.cryo2d_closecopy import (
    DEFAULT_CLOSECOPY_SEEDS,
    U2D,
    grad_U2D,
    find_1ion_planar_minimum,
    make_triangular_seed,
)


class HarmonicTrapOnlySim:
    def __init__(self, *, kx=0.55, ky=2.5, kz=1.6):
        self.kx = float(kx)
        self.ky = float(ky)
        self.kz = float(kz)
        self.trapVariables = SimpleNamespace(dc_key="dc")
        self.center_fits = {}

    def _ensure_dc_center_fit(self, polyfit_deg=4):
        return None

    def evaluate_center_poly_2ndderivatives(self, x, y, z):
        return np.array(
            [
                [self.kx, 0.0, 0.0],
                [0.0, self.ky, 0.0],
                [0.0, 0.0, self.kz],
            ],
            dtype=float,
        )

    def get_U_using_polyfit_dimensionless(self, ionpos_flat):
        positions = np.asarray(ionpos_flat, dtype=float).reshape(-1, 3)
        return 0.5 * np.sum(
            self.kx * positions[:, 0] ** 2
            + self.ky * positions[:, 1] ** 2
            + self.kz * positions[:, 2] ** 2
        )

    def get_U_Grad_using_polyfit_dimensionless(self, ionpos_flat):
        positions = np.asarray(ionpos_flat, dtype=float).reshape(-1, 3)
        grad = np.zeros_like(positions)
        grad[:, 0] = self.kx * positions[:, 0]
        grad[:, 1] = self.ky * positions[:, 1]
        grad[:, 2] = self.kz * positions[:, 2]
        return grad.reshape(-1)


def test_default_closecopy_seed_specs_are_exactly_three_pairs():
    assert len(DEFAULT_CLOSECOPY_SEEDS) == 3
    assert all(len(seed_spec) == 2 for seed_spec in DEFAULT_CLOSECOPY_SEEDS)
    assert DEFAULT_CLOSECOPY_SEEDS == ((0.9, 11), (1.0, 11), (1.1, 11))


def test_make_triangular_seed_centers_com_and_matches_defaults():
    seed = make_triangular_seed(9, 1.8, spacing=1.0, jitter=0.02, rng_seed=11)
    assert seed.spacing == 1.0
    assert seed.jitter == 0.02
    assert seed.rng_seed == 11
    np.testing.assert_allclose(seed.points_2d.mean(axis=0), np.zeros(2), atol=1.0e-14)
    np.testing.assert_allclose(seed.uv0[:9], seed.u0)
    np.testing.assert_allclose(seed.uv0[9:], seed.v0)


def test_find_1ion_planar_minimum_is_origin_for_harmonic_trap():
    sim = HarmonicTrapOnlySim(kx=0.5, ky=3.0, kz=2.0)
    context = find_1ion_planar_minimum(sim)
    np.testing.assert_allclose(context.center_xz_dimless, np.zeros(2), atol=1.0e-12)
    np.testing.assert_allclose(context.center_3d_dimless, np.zeros(3), atol=1.0e-12)
    assert np.isclose(context.alpha_seed, 2.0)


def test_grad_u2d_projects_out_com_mode():
    sim = HarmonicTrapOnlySim(kx=1.0, ky=5.0, kz=4.0)
    center = np.zeros(2, dtype=float)
    uv = np.array([1.0, -0.5, 0.25, 0.2, -0.1, 0.6], dtype=float)
    energy = U2D(uv, sim, center)
    grad = grad_U2D(uv, sim, center)
    assert np.isfinite(energy)
    assert grad.shape == uv.shape
    assert np.isclose(np.sum(grad[:3]), 0.0)
    assert np.isclose(np.sum(grad[3:]), 0.0)
