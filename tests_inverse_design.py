import numpy as np

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
    A = np.eye(M, dtype=float)

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
            rf_freq_hz=1.0,
            num_samples=2,
            objective="l2",
            enforce_bounds=False,
            u_bounds=[(-1.0, 1.0)] * M,
        )
        u = out["u"]
        # With A=I, u is the coefficient vector. Check L u == b.
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
    A = rng.standard_normal((M, K))
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
            rf_freq_hz=1.0,
            num_samples=2,
            objective="l2",
            enforce_bounds=False,
            u_bounds=[(-1.0, 1.0)] * K,
        )
        np.testing.assert_allclose(out["u"], u_star, rtol=0, atol=1e-9)
    finally:
        idesign.build_voltage_to_c_matrix = orig

