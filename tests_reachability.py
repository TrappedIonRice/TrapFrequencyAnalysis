import numpy as np
import pytest

import constants
import reachability.model as reachability_model_module
from control_constraints import build_derivative_rows, build_hessian_rows
from reachability import (
    build_modal_curvature_hull,
    ReachabilityModel,
    build_reachability_model,
    deduplicate_modal_curvature_points,
    sample_reachable_boundary,
    solve_reachability_support_query,
)


def _mock_powers_and_A():
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
    rng = np.random.default_rng(1234)
    A = rng.standard_normal((powers.shape[0], 4))
    return powers, A


def _build_model_with_mock_A(monkeypatch, **builder_kwargs):
    powers, A = _mock_powers_and_A()

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    monkeypatch.setattr(reachability_model_module, "build_voltage_to_c_matrix", fake_builder)
    r0 = np.array([0.15e-6, -0.2e-6, 0.05e-6], dtype=float)
    model = build_reachability_model(
        r0=r0,
        principal_axis=np.array([1.0, 0.0, 0.0]),
        ref_dir=np.array([0.0, 1.0, 0.0]),
        alpha_deg=0.0,
        trap_name="T",
        dc_electrodes=["DC1", "DC2"],
        rf_dc_electrodes=["RF1"],
        num_samples=3,
        use_cache=False,
        u_bounds=[(-2.0, 2.0), (-1.0, 1.0), (-0.5, 0.5), (0.0, 3.0)],
        ion_mass_kg=constants.ion_mass,
        **builder_kwargs,
    )
    return model, powers, A, r0


def _make_toy_model(lower, upper):
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    u_bounds = [(float(a), float(b)) for a, b in zip(lo, hi)]
    return ReachabilityModel(
        A=np.eye(3, dtype=float),
        powers=np.zeros((3, 3), dtype=int),
        L_eq_c=np.zeros((0, 3), dtype=float),
        b_eq=np.zeros(0, dtype=float),
        L_diag_c=np.eye(3, dtype=float),
        E=np.zeros((0, 3), dtype=float),
        e=np.zeros(0, dtype=float),
        T=np.eye(3, dtype=float),
        lower_u=lo,
        upper_u=hi,
        u_bounds=u_bounds,
        r0=np.zeros(3, dtype=float),
        rotation=np.eye(3, dtype=float),
        metadata={"toy": True},
    )


def test_reachability_model_construction_matches_modal_diagonal_pushforward(monkeypatch):
    model, powers, _, r0 = _build_model_with_mock_A(monkeypatch)
    np.testing.assert_allclose(model.T, model.L_diag_c @ model.A, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.lower_u, np.array([-2.0, -1.0, -0.5, 0.0]), rtol=0, atol=0)
    np.testing.assert_allclose(model.upper_u, np.array([2.0, 1.0, 0.5, 3.0]), rtol=0, atol=0)

    r_eval = r0 / constants.ND_L0_M
    dxx, dyy, dzz, _, _, _ = build_hessian_rows(powers, r_eval)
    np.testing.assert_allclose(model.L_diag_c[0], dxx, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.L_diag_c[1], dyy, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.L_diag_c[2], dzz, rtol=0, atol=1e-12)


def test_reachability_model_equality_map_structure(monkeypatch):
    model, powers, _, r0 = _build_model_with_mock_A(monkeypatch)
    r_eval = r0 / constants.ND_L0_M
    dx, dy, dz = build_derivative_rows(powers, r_eval)
    _, _, _, dxy, dxz, dyz = build_hessian_rows(powers, r_eval)
    expected = np.vstack([dx, dy, dz, dxy, dxz, dyz]).astype(float)

    np.testing.assert_allclose(model.L_eq_c, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.b_eq, np.zeros(6), rtol=0, atol=0)
    np.testing.assert_allclose(model.E, expected @ model.A, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.e, model.b_eq, rtol=0, atol=0)


def test_support_query_toy_system_returns_expected_boundary_point():
    model = _make_toy_model(lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0])
    direction = np.array([1.0, -2.0, 0.5], dtype=float)
    out = solve_reachability_support_query(model, direction)

    assert out.success is True
    np.testing.assert_allclose(out.u_star, np.array([1.0, 0.0, 1.0]), rtol=0, atol=1e-9)
    np.testing.assert_allclose(out.lambda_star, np.array([1.0, 0.0, 1.0]), rtol=0, atol=1e-9)
    np.testing.assert_allclose(out.objective_value, direction @ out.lambda_star, rtol=0, atol=1e-12)


def test_support_query_zero_direction_raises():
    model = _make_toy_model(lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        solve_reachability_support_query(model, np.zeros(3))


def test_sampling_wrapper_returns_requested_sample_count():
    model = _make_toy_model(lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0])
    n_samples = 37
    out = sample_reachable_boundary(
        model,
        n_samples=n_samples,
        random_seed=7,
        deduplicate_tol=None,
        build_hull=False,
    )

    assert out.n_requested == n_samples
    assert len(out.query_results) == n_samples
    assert out.sampled_directions.shape == (n_samples, 3)
    assert out.n_success == n_samples
    assert out.n_raw_returned == n_samples
    assert out.raw_lambda_points.shape == (n_samples, 3)
    assert out.lambda_points.shape == (n_samples, 3)
    assert out.hull is None
    np.testing.assert_allclose(
        np.linalg.norm(out.sampled_directions, axis=1),
        np.ones(n_samples),
        rtol=0,
        atol=1e-12,
    )


def test_zero_curvature_is_feasible_in_zero_centered_toy_setup():
    model = _make_toy_model(lower=[-1.0, -1.0, -1.0], upper=[1.0, 1.0, 1.0])
    u0 = np.zeros(3, dtype=float)
    assert np.all(u0 >= model.lower_u)
    assert np.all(u0 <= model.upper_u)
    np.testing.assert_allclose(model.E @ u0, model.e, rtol=0, atol=0)
    np.testing.assert_allclose(model.T @ u0, np.zeros(3), rtol=0, atol=0)


def test_deduplicate_points_collapses_within_tolerance():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0e-7, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    unique, keep = deduplicate_modal_curvature_points(pts, tol=1.0e-6)
    assert unique.shape == (2, 3)
    np.testing.assert_array_equal(keep, np.array([0, 2], dtype=int))


def test_deduplicate_points_keeps_points_outside_tolerance():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0e-6, 0.0, 0.0],
        ],
        dtype=float,
    )
    unique, keep = deduplicate_modal_curvature_points(pts, tol=1.0e-6)
    assert unique.shape == (2, 3)
    np.testing.assert_array_equal(keep, np.array([0, 1], dtype=int))


def test_convex_hull_builds_for_simple_tetrahedron():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0e-10, 0.0, 0.0],
        ],
        dtype=float,
    )
    out = build_modal_curvature_hull(pts, deduplicate_tol=1.0e-9)
    assert out.status == "ok"
    assert out.hull is not None
    assert out.n_input_points == 5
    assert out.n_unique_points == 4
    assert out.hull_vertices.shape[0] == 4
    assert out.hull_simplices.shape[1] == 3


def test_convex_hull_handles_low_point_count_gracefully():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    out = build_modal_curvature_hull(pts, deduplicate_tol=1.0e-9)
    assert out.status == "insufficient_points"
    assert out.hull is None
    assert out.hull_simplices.shape == (0, 3)


def test_reachability_model_rejects_mass_mismatch(monkeypatch):
    powers, A = _mock_powers_and_A()

    def fake_builder(**kwargs):
        return {"A": A, "powers": powers}

    monkeypatch.setattr(reachability_model_module, "build_voltage_to_c_matrix", fake_builder)
    with pytest.raises(ValueError, match="mass mismatch"):
        build_reachability_model(
            r0=np.array([0.0, 0.0, 0.0]),
            principal_axis=np.array([1.0, 0.0, 0.0]),
            ref_dir=np.array([0.0, 1.0, 0.0]),
            alpha_deg=0.0,
            trap_name="T",
            dc_electrodes=["DC1", "DC2"],
            rf_dc_electrodes=["RF1"],
            num_samples=3,
            use_cache=False,
            u_bounds=[(-2.0, 2.0), (-1.0, 1.0), (-0.5, 0.5), (0.0, 3.0)],
            ion_mass_kg=constants.ion_mass * 1.1,
        )


def test_reserved_reachability_builder_args_are_noop(monkeypatch):
    model_default, _, _, _ = _build_model_with_mock_A(monkeypatch)
    model_reserved, _, _, _ = _build_model_with_mock_A(
        monkeypatch,
        ion_charge_c=7.5,
        poly_is_potential_energy=False,
        freqs_in_hz=True,
    )
    np.testing.assert_allclose(model_default.E, model_reserved.E, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model_default.T, model_reserved.T, rtol=0, atol=1e-12)
    assert "api_scope_note" in model_reserved.metadata
    assert model_reserved.metadata["ion_charge_c_reserved"] == 7.5
    assert model_reserved.metadata["poly_is_potential_energy_reserved"] is False
    assert model_reserved.metadata["freqs_in_hz_reserved"] is True
