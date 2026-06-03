import numpy as np

from equilibrium_playground.symmetry_enforced_harmonic2d import normalize_harmonic_trap
from equilibrium_playground.unrestricted_harmonic2d import (
    sorted_rms_displacement,
    unrestricted_total_energy,
    unrestricted_total_gradient,
)


def test_unrestricted_total_gradient_matches_finite_difference():
    trap = normalize_harmonic_trap(alpha=1.4, scale=0.9)
    q = np.array([0.2, -0.45, 0.7, 0.15, -0.25, 0.4], dtype=float)

    analytic = unrestricted_total_gradient(q, trap)
    numeric = np.zeros_like(q)
    eps = 1.0e-7
    for idx in range(q.size):
        step = np.zeros_like(q)
        step[idx] = eps
        numeric[idx] = (
            unrestricted_total_energy(q + step, trap)
            - unrestricted_total_energy(q - step, trap)
        ) / (2.0 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1.0e-6, atol=1.0e-7)


def test_sorted_rms_displacement_is_permutation_insensitive():
    points_a = np.array(
        [
            [0.0, 0.0],
            [1.0, -0.5],
            [-0.75, 0.25],
        ],
        dtype=float,
    )
    points_b = np.array(
        [
            [1.0, -0.5],
            [-0.75, 0.25],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    assert np.isclose(sorted_rms_displacement(points_a, points_b), 0.0)
