import numpy as np
import constants
import pytest

from control_constraints import build_L_b_for_point, rotation_from_axis_ref_alpha


def test_quadratic_hessian_matches():
    # U = a x^2 + b y^2 + c z^2 + d xy
    a, b, c, d = 1.2, -0.5, 0.8, 0.3
    powers = np.array(
        [
            [2, 0, 0],  # x^2
            [0, 2, 0],  # y^2
            [0, 0, 2],  # z^2
            [1, 1, 0],  # xy
        ],
        dtype=int,
    )
    coeffs = np.array([a, b, c, d], dtype=float)
    L0 = constants.ND_L0_M
    degrees = np.sum(powers, axis=1)
    coeffs_nd = coeffs * (L0 ** degrees)
    r0 = np.array([0.1, -0.2, 0.05], dtype=float)

    Kstar = np.array(
        [
            [2 * a, d, 0.0],
            [d, 2 * b, 0.0],
            [0.0, 0.0, 2 * c],
        ],
        dtype=float,
    )

    L, b = build_L_b_for_point(powers, r0, Kstar, include_gradient=False)
    np.testing.assert_allclose(L @ coeffs_nd, b, rtol=0, atol=1e-12)


def test_gradient_rows_pick_linear_terms():
    # U = ax + by + cz
    a, b, c = 0.7, -1.1, 2.2
    powers = np.array(
        [
            [1, 0, 0],  # x
            [0, 1, 0],  # y
            [0, 0, 1],  # z
        ],
        dtype=int,
    )
    coeffs = np.array([a, b, c], dtype=float)
    L0 = constants.ND_L0_M
    degrees = np.sum(powers, axis=1)
    coeffs_nd = coeffs * (L0 ** degrees)
    r0 = np.array([0.3, -0.4, 0.5], dtype=float)
    Kstar = np.zeros((3, 3), dtype=float)

    L, b = build_L_b_for_point(powers, r0, Kstar, include_gradient=True)
    grad = (L @ coeffs_nd)[:3]
    np.testing.assert_allclose(grad, np.array([a * L0, b * L0, c * L0], dtype=float), rtol=0, atol=1e-12)
    np.testing.assert_allclose(b[:3], np.zeros(3), rtol=0, atol=0)


def test_hessian_vech_ordering():
    # U = d xy should map to vech entry [xx, yy, zz, xy, xz, yz]
    d = 1.7
    powers = np.array([[1, 1, 0]], dtype=int)
    coeffs = np.array([d], dtype=float)
    L0 = constants.ND_L0_M
    coeffs_nd = coeffs * (L0 ** 2)
    r0 = np.array([0.0, 0.0, 0.0], dtype=float)
    Kstar = np.zeros((3, 3), dtype=float)

    L, _ = build_L_b_for_point(powers, r0, Kstar, include_gradient=False)
    vech = L @ coeffs_nd
    expected = np.array([0.0, 0.0, 0.0, d * (L0 ** 2), 0.0, 0.0], dtype=float)
    np.testing.assert_allclose(vech, expected, rtol=0, atol=1e-12)


def test_rotation_alpha_sign_convention():
    axis = np.array([1.0, 0.0, 0.0])
    ref = np.array([0.0, 1.0, 0.0])
    R = rotation_from_axis_ref_alpha(axis, ref, 90.0)
    e1, e2, e3 = R[:, 0], R[:, 1], R[:, 2]
    np.testing.assert_allclose(e1, np.array([1.0, 0.0, 0.0]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(e2, np.array([0.0, 0.0, 1.0]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(e3, np.array([0.0, -1.0, 0.0]), rtol=0, atol=1e-12)


def test_rotation_degenerate_ref_raises():
    axis = np.array([0.0, 0.0, 1.0])
    ref = np.array([0.0, 0.0, 2.0])
    with pytest.raises(ValueError):
        rotation_from_axis_ref_alpha(axis, ref, 0.0)


def test_rotation_axis_warning(capsys):
    axis = np.array([2.0, 0.0, 0.0])  # norm 2 -> warn
    ref = np.array([0.0, 1.0, 0.0])
    rotation_from_axis_ref_alpha(axis, ref, 0.0)
    captured = capsys.readouterr()
    assert "Warning" in captured.out
