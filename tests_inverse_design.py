import numpy as np
import pytest
import constants

import inverse_design as idesign
from control_constraints import build_L_b_for_point, build_target_hessian


def test_solve_l2_identity_A():
    # Use A = I so u == c. We monkeypatch the A-builder.
    powers = np.array(
        [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=int,
    )
    M = powers.shape[0]
    n_u = M + 1
    A_phys = np.zeros((M, n_u), dtype=float)
    A_phys[:, :M] = np.eye(M, dtype=float)
    L0 = constants.ND_L0_M
    degrees = np.sum(powers, axis=1)
    scale = (L0 ** degrees).astype(float)
    A = scale[:, None] * A_phys

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    orig = idesign.build_voltage_to_c_matrix
    idesign.build_voltage_to_c_matrix = fake_builder
    try:
        r0 = np.array([0.0, 0.0, 0.0], dtype=float)
        freqs = np.array([1e6, 1.2e6, 0.8e6], dtype=float)
        R = np.eye(3)
        out = idesign.solve_u_for_targets(
            r0=r0,
            freqs=freqs,
            principal_dirs=R,
            ion_mass_kg=1.0,
            ion_charge_c=1.0,
            poly_is_potential_energy=False,
            freqs_in_hz=True,
            trap_name="T",
            dc_electrodes=["DC1"],
            rf_dc_electrodes=["RF1"],
            num_samples=2,
            s_bounds=(0.0, 1.0),
            objective="l2",
            enforce_bounds=False,
        )
        u = out["u"]
        # With A=[I 0], first M entries are the coefficient vector.
        np.testing.assert_allclose(out["M"] @ u, out["b"], rtol=0, atol=1e-9)
    finally:
        idesign.build_voltage_to_c_matrix = orig


def test_min_norm_matches_closed_form():
    # Random small system with A-builder mocked.
    rng = np.random.default_rng(0)
    powers = np.array(
        [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=int,
    )
    M = powers.shape[0]
    K = 4
    K_rf = 2
    n_u = K + K_rf + 1
    A_phys = rng.standard_normal((M, n_u))
    L0 = constants.ND_L0_M
    degrees = np.sum(powers, axis=1)
    scale = (L0 ** degrees).astype(float)
    A = scale[:, None] * A_phys
    r0 = np.array([0.1, -0.2, 0.05], dtype=float)
    freqs = np.array([1.0, 1.2, 0.8], dtype=float)
    R = np.eye(3)
    Kstar = build_target_hessian(freqs, R, mass=1.0, charge=1.0, poly_is_potential_energy=False)
    L, b = build_L_b_for_point(powers, r0, Kstar, include_gradient=True)
    Mmat = L @ A

    # Closed-form min-norm solution
    MMt = Mmat @ Mmat.T
    x = np.linalg.solve(MMt, b)
    u_star = Mmat.T @ x

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    orig = idesign.build_voltage_to_c_matrix
    idesign.build_voltage_to_c_matrix = fake_builder
    try:
        out = idesign.solve_u_for_targets(
            r0=r0,
            freqs=freqs,
            principal_dirs=R,
            ion_mass_kg=1.0,
            ion_charge_c=1.0,
            poly_is_potential_energy=False,
            freqs_in_hz=False,
            trap_name="T",
            dc_electrodes=["DC1"],
            rf_dc_electrodes=["RF1", "RF2"],
            num_samples=2,
            s_bounds=(0.0, 1.0),
            objective="l2",
            enforce_bounds=False,
        )
        np.testing.assert_allclose(out["u"], u_star, rtol=0, atol=1e-9)
    finally:
        idesign.build_voltage_to_c_matrix = orig


def test_unbounded_infeasible_returns_best_effort():
    # Mmat = 0, b != 0 => infeasible; should return best-effort u (not None)
    powers = np.array([[2, 0, 0]], dtype=int)
    A = np.zeros((1, 2), dtype=float)

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    orig = idesign.build_voltage_to_c_matrix
    idesign.build_voltage_to_c_matrix = fake_builder
    try:
        out = idesign.solve_u_for_targets(
            r0=np.array([0.0, 0.0, 0.0]),
            freqs=np.array([1.0, 1.0, 1.0]),
            principal_dirs=np.eye(3),
            ion_mass_kg=1.0,
            ion_charge_c=1.0,
            poly_is_potential_energy=False,
            freqs_in_hz=False,
            trap_name="T",
            dc_electrodes=["DC1"],
            rf_dc_electrodes=[],
            num_samples=2,
            s_bounds=(0.0, 1.0),
            objective="l2",
            enforce_bounds=False,
        )
        assert out["u"] is not None
        assert out["status"] == "best_effort_infeasible"
        assert out["solver_info"]["resid_ls"] > out["solver_info"]["eq_tol"]
    finally:
        idesign.build_voltage_to_c_matrix = orig


def test_bounded_infeasible_returns_none():
    if not idesign._HAVE_SCIPY:
        pytest.skip("SciPy not available for constrained test")

    powers = np.array([[2, 0, 0]], dtype=int)
    A = np.zeros((1, 2), dtype=float)

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    orig = idesign.build_voltage_to_c_matrix
    idesign.build_voltage_to_c_matrix = fake_builder
    try:
        out = idesign.solve_u_for_targets(
            r0=np.array([0.0, 0.0, 0.0]),
            freqs=np.array([1.0, 1.0, 1.0]),
            principal_dirs=np.eye(3),
            ion_mass_kg=1.0,
            ion_charge_c=1.0,
            poly_is_potential_energy=False,
            freqs_in_hz=False,
            trap_name="T",
            dc_electrodes=["DC1"],
            rf_dc_electrodes=[],
            num_samples=2,
            s_bounds=(0.0, 1.0),
            objective="l2",
            enforce_bounds=True,
            u_bounds=[(0.0, 0.0), (0.0, 0.0)],
        )
        assert out["u"] is None
        assert out["status"] in ("infeasible_bounds", "solver_failed")
    finally:
        idesign.build_voltage_to_c_matrix = orig
