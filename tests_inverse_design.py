import numpy as np
import pytest
import constants

import inverse_design as idesign
from control_constraints import build_L_b_for_point, build_target_hessian


def test_solve_l2_identity_A():
    # Use A = I so u == c. We monkeypatch the A-builder.
    # use_cache=False ensures the monkeypatched builder is actually exercised.
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
        out = idesign.solve_u_for_targets(
            r0=r0,
            freqs=freqs,
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
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
            use_cache=False,
        )
        u = out["u"]
        # With A=[I 0], first M entries are the coefficient vector.
        np.testing.assert_allclose(out["M"] @ u, out["b"], rtol=0, atol=1e-9)
    finally:
        idesign.build_voltage_to_c_matrix = orig


def test_min_norm_matches_closed_form():
    # Random small system with A-builder mocked.
    # use_cache=False ensures the monkeypatched builder is actually exercised.
    rng = np.random.default_rng(0)
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
    K = 7
    K_rf = 4
    n_u = K + K_rf + 1
    A_phys = rng.standard_normal((M, n_u))
    L0 = constants.ND_L0_M
    degrees = np.sum(powers, axis=1)
    scale = (L0 ** degrees).astype(float)
    A = scale[:, None] * A_phys
    r0 = np.array([0.1, -0.2, 0.05], dtype=float)
    freqs = np.array([1.0, 1.2, 0.8], dtype=float)
    Kstar = build_target_hessian(
        freqs,
        principal_axis=np.array([1.0, 0.0, 0.0]),
        ref_dir=np.array([0.0, 1.0, 0.0]),
        alpha_deg=0.0,
        mass=1.0,
        charge=1.0,
        poly_is_potential_energy=False,
    )
    L, b = build_L_b_for_point(powers, r0, Kstar, include_gradient=True)
    Mmat = L @ A

    # Ensure full-row-rank so the closed-form path is unambiguous.
    tries = 0
    while np.linalg.matrix_rank(Mmat) < Mmat.shape[0]:
        A_phys = rng.standard_normal((M, n_u))
        A = scale[:, None] * A_phys
        Mmat = L @ A
        tries += 1
        if tries > 20:
            raise RuntimeError("Could not build full-row-rank test system")

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
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
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
            use_cache=False,
        )
        np.testing.assert_allclose(out["u"], u_star, rtol=0, atol=1e-9)
    finally:
        idesign.build_voltage_to_c_matrix = orig


def test_exact_mode_returns_legacy_constraint_rows():
    # Exact mode should preserve legacy returned L,b,M semantics/order.
    # use_cache=False ensures the monkeypatched builder is actually exercised.
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
    A = np.zeros((M, n_u), dtype=float)
    A[:, :M] = np.eye(M, dtype=float)

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    orig = idesign.build_voltage_to_c_matrix
    idesign.build_voltage_to_c_matrix = fake_builder
    try:
        r0 = np.array([0.0, 0.0, 0.0], dtype=float)
        freqs = np.array([1.0e6, 1.2e6, 0.8e6], dtype=float)
        out = idesign.solve_u_for_targets(
            r0=r0,
            freqs=freqs,
            target_mode="exact",
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
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
            use_cache=False,
        )
        Kstar = build_target_hessian(
            freqs,
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
            mass=1.0,
            charge=1.0,
            poly_is_potential_energy=False,
            freqs_in_hz=True,
        )
        L_expected, b_expected = build_L_b_for_point(powers, r0, Kstar, include_gradient=True)
        np.testing.assert_allclose(out["L"], L_expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(out["b"], b_expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(out["M"], L_expected @ A, rtol=0, atol=1e-12)
    finally:
        idesign.build_voltage_to_c_matrix = orig


def test_unbounded_infeasible_returns_best_effort():
    # Mmat = 0, b != 0 => infeasible; should return best-effort u (not None)
    # use_cache=False ensures the monkeypatched builder is actually exercised.
    powers = np.array([[2, 0, 0]], dtype=int)
    A = np.zeros((1, 2), dtype=float)

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    orig = idesign.build_voltage_to_c_matrix
    idesign.build_voltage_to_c_matrix = fake_builder
    try:
        out = idesign.solve_u_for_targets(
            r0=np.array([0.0, 0.0, 0.0]),
            freqs=np.array([1.0e6, 1.0e6, 1.0e6]),
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
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
            use_cache=False,
        )
        assert out["u"] is not None
        assert out["status"] == "best_effort_infeasible"
        assert out["solver_info"]["resid_ls"] > out["solver_info"]["eq_tol"]
    finally:
        idesign.build_voltage_to_c_matrix = orig


def test_bounded_infeasible_returns_none():
    if not idesign._HAVE_SCIPY:
        pytest.skip("SciPy not available for constrained test")

    # use_cache=False ensures the monkeypatched builder is actually exercised.
    powers = np.array([[2, 0, 0]], dtype=int)
    A = np.zeros((1, 2), dtype=float)

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    orig = idesign.build_voltage_to_c_matrix
    idesign.build_voltage_to_c_matrix = fake_builder
    try:
        out = idesign.solve_u_for_targets(
            r0=np.array([0.0, 0.0, 0.0]),
            freqs=np.array([1.0e6, 1.0e6, 1.0e6]),
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
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
            use_cache=False,
        )
        assert out["u"] is None
        assert out["status"] in ("infeasible_bounds", "solver_failed")
    finally:
        idesign.build_voltage_to_c_matrix = orig


def test_l2_with_linear_inequalities_feasible():
    if not idesign._HAVE_SCIPY:
        pytest.skip("SciPy not available for constrained test")
    # use_cache=False ensures the monkeypatched builder is actually exercised.

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
    A = np.zeros((M, n_u), dtype=float)
    A[:, :M] = np.eye(M, dtype=float)

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    orig = idesign.build_voltage_to_c_matrix
    idesign.build_voltage_to_c_matrix = fake_builder
    try:
        kwargs = dict(
            r0=np.array([0.0, 0.0, 0.0], dtype=float),
            freqs=np.array([1.0e6, 1.2e6, 0.8e6], dtype=float),
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
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
            use_cache=False,
        )
        out_ref = idesign.solve_u_for_targets(**kwargs)
        u_ref = out_ref["u"]
        assert u_ref is not None

        eps = 1e-7
        G = np.zeros((2, n_u), dtype=float)
        G[0, 0] = 1.0
        G[1, 0] = -1.0
        h = np.array([u_ref[0] + eps, -(u_ref[0] - eps)], dtype=float)
        out = idesign.solve_u_for_targets(**kwargs, G_ub=G, h_ub=h)
        assert out["status"] == "ok"
        assert out["u"] is not None
        assert out["solver_info"]["ineq_violation"] <= out["solver_info"]["ineq_tol"]
        assert out["solver_info"]["ineq_violation_rel"] <= 1e-12
    finally:
        idesign.build_voltage_to_c_matrix = orig


def test_l2_with_linear_inequalities_infeasible():
    if not idesign._HAVE_SCIPY:
        pytest.skip("SciPy not available for constrained test")
    # use_cache=False ensures the monkeypatched builder is actually exercised.

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
    A = np.zeros((M, n_u), dtype=float)
    A[:, :M] = np.eye(M, dtype=float)

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    orig = idesign.build_voltage_to_c_matrix
    idesign.build_voltage_to_c_matrix = fake_builder
    try:
        kwargs = dict(
            r0=np.array([0.0, 0.0, 0.0], dtype=float),
            freqs=np.array([1.0e6, 1.2e6, 0.8e6], dtype=float),
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
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
            use_cache=False,
        )
        out_ref = idesign.solve_u_for_targets(**kwargs)
        u_ref = out_ref["u"]
        assert u_ref is not None

        G = np.zeros((2, n_u), dtype=float)
        G[0, 0] = 1.0
        G[1, 0] = -1.0
        h = np.array([u_ref[0] - 0.5, -(u_ref[0] + 0.5)], dtype=float)
        out = idesign.solve_u_for_targets(**kwargs, G_ub=G, h_ub=h)
        assert out["u"] is None
        assert out["status"] in ("infeasible_bounds", "solver_failed")
        assert out["solver_info"]["ineq_violation"] > out["solver_info"]["ineq_tol"]
    finally:
        idesign.build_voltage_to_c_matrix = orig


def test_linf_with_linear_inequalities_feasible():
    if not idesign._HAVE_SCIPY:
        pytest.skip("SciPy not available for LP constrained test")
    # use_cache=False ensures the monkeypatched builder is actually exercised.

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
    A = np.zeros((M, n_u), dtype=float)
    A[:, :M] = np.eye(M, dtype=float)

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    orig = idesign.build_voltage_to_c_matrix
    idesign.build_voltage_to_c_matrix = fake_builder
    try:
        kwargs = dict(
            r0=np.array([0.0, 0.0, 0.0], dtype=float),
            freqs=np.array([0.9e6, 1.1e6, 1.3e6], dtype=float),
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
            ion_mass_kg=1.0,
            ion_charge_c=1.0,
            poly_is_potential_energy=False,
            freqs_in_hz=True,
            trap_name="T",
            dc_electrodes=["DC1"],
            rf_dc_electrodes=["RF1"],
            num_samples=2,
            s_bounds=(0.0, 1.0),
            objective="linf",
            enforce_bounds=False,
            use_cache=False,
        )
        out_ref = idesign.solve_u_for_targets(**kwargs)
        u_ref = out_ref["u"]
        assert u_ref is not None
        G = np.zeros((1, n_u), dtype=float)
        G[0, 0] = 1.0
        h = np.array([u_ref[0] + 0.2], dtype=float)
        out = idesign.solve_u_for_targets(**kwargs, G_ub=G, h_ub=h)
        assert out["status"] == "ok"
        assert out["u"] is not None
        assert out["solver_info"]["ineq_violation"] <= out["solver_info"]["ineq_tol"]
    finally:
        idesign.build_voltage_to_c_matrix = orig


def test_ineq_tol_scales_with_rhs_magnitude():
    t_small = idesign._ineq_tol(np.array([1.0, -2.0], dtype=float))
    t_big = idesign._ineq_tol(np.array([1.0e6, -2.0e6], dtype=float))
    assert t_big > t_small


def test_frequency_box_wrapper_feasible():
    if not idesign._HAVE_SCIPY:
        pytest.skip("SciPy not available for constrained test")
    # use_cache=False ensures the monkeypatched builder is actually exercised.

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
    A = np.zeros((M, n_u), dtype=float)
    A[:, :M] = np.eye(M, dtype=float)

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    orig = idesign.build_voltage_to_c_matrix
    idesign.build_voltage_to_c_matrix = fake_builder
    try:
        exact = idesign.solve_u_for_exact_targets(
            r0=np.array([0.0, 0.0, 0.0], dtype=float),
            freqs=np.array([1.0e6, 1.2e6, 0.8e6], dtype=float),
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
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
            use_cache=False,
        )
        assert exact["status"] == "ok"
        assert exact["u"] is not None

        out = idesign.solve_u_for_frequency_box(
            r0=np.array([0.0, 0.0, 0.0], dtype=float),
            freq_bounds=[(0.9e6, 1.1e6), (1.1e6, 1.3e6), (0.7e6, 0.9e6)],
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
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
            use_cache=False,
        )
        assert out["status"] == "ok"
        assert out["u"] is not None
        assert out["L"].shape[0] == 6  # grad(3) + offdiag modal equalities(3)
        assert out["solver_info"]["ineq_violation"] <= out["solver_info"]["ineq_tol"]
    finally:
        idesign.build_voltage_to_c_matrix = orig


def test_frequency_box_infeasible_returns_none():
    if not idesign._HAVE_SCIPY:
        pytest.skip("SciPy not available for constrained test")
    # use_cache=False ensures the monkeypatched builder is actually exercised.

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
    A = np.zeros((powers.shape[0], 2), dtype=float)

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    orig = idesign.build_voltage_to_c_matrix
    idesign.build_voltage_to_c_matrix = fake_builder
    try:
        out = idesign.solve_u_for_frequency_box(
            r0=np.array([0.0, 0.0, 0.0], dtype=float),
            freq_bounds=[(0.5e6, None), (0.5e6, None), (0.5e6, None)],
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
            ion_mass_kg=1.0,
            ion_charge_c=1.0,
            poly_is_potential_energy=False,
            freqs_in_hz=True,
            trap_name="T",
            dc_electrodes=["DC1"],
            rf_dc_electrodes=[],
            num_samples=2,
            objective="l2",
            enforce_bounds=False,
            use_cache=False,
        )
        assert out["u"] is None
        assert out["status"] in ("infeasible_bounds", "solver_failed")
    finally:
        idesign.build_voltage_to_c_matrix = orig


def test_frequency_box_validation_errors():
    with pytest.raises(ValueError):
        idesign.solve_u_for_targets(
            r0=np.array([0.0, 0.0, 0.0], dtype=float),
            freqs=None,
            freq_bounds=[(1.0e6, 0.9e6), (1.0e6, 1.1e6), (1.0e6, 1.1e6)],
            target_mode="box",
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
            ion_mass_kg=1.0,
            ion_charge_c=1.0,
            poly_is_potential_energy=False,
            freqs_in_hz=True,
            trap_name="T",
            dc_electrodes=["DC1"],
            rf_dc_electrodes=[],
            num_samples=2,
            objective="l2",
            enforce_bounds=False,
            use_cache=False,
        )
