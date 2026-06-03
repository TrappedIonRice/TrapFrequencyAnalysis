import numpy as np
from types import SimpleNamespace

import equilibrium_playground.symmetry_scan as symmetry_scan
from equilibrium_playground.symmetry_scan import (
    SymmetryFamilyScanResult,
    best_of_k_symmetry_enforced_harmonic2d,
    build_alpha_grid,
    build_num_ions_free_grid,
    compute_median_energy_by_total_ion_count,
    compute_energy_residual_surface,
    fit_energy_n_trend_power_law,
    fit_normalized_energy_baseline_polynomial,
    normalize_scan_energy_by_fitted_n_trend,
    normalize_scan_energy_by_n2,
)


def _make_stub_scan_result(
    *,
    num_ions_vert: int,
    num_ions_hor: int,
    alpha_grid: np.ndarray,
    num_ions_free_grid: np.ndarray,
    energy_surface: np.ndarray,
    success_mask: np.ndarray,
    total_ion_count_surface: np.ndarray,
) -> SymmetryFamilyScanResult:
    shape = energy_surface.shape
    return SymmetryFamilyScanResult(
        num_ions_vert=num_ions_vert,
        num_ions_hor=num_ions_hor,
        scale=1.0,
        best_of_k=1,
        solver_settings={},
        alpha_grid=np.asarray(alpha_grid, dtype=float),
        num_ions_free_grid=np.asarray(num_ions_free_grid, dtype=int),
        point_results=np.empty(shape, dtype=object),
        energy_surface=np.asarray(energy_surface, dtype=float),
        success_mask=np.asarray(success_mask, dtype=bool),
        runtime_surface_s=np.zeros(shape, dtype=float),
        gradient_norm_surface=np.zeros(shape, dtype=float),
        total_ion_count_surface=np.asarray(total_ion_count_surface, dtype=int),
        message_surface=np.empty(shape, dtype=object),
    )


def test_build_alpha_grid_is_inclusive_and_linear():
    alpha_grid = build_alpha_grid(1.0, 2.0, 5)
    np.testing.assert_allclose(alpha_grid, np.array([1.0, 1.25, 1.5, 1.75, 2.0]))


def test_build_num_ions_free_grid_is_inclusive():
    free_grid = build_num_ions_free_grid(2, 6)
    np.testing.assert_array_equal(free_grid, np.array([2, 3, 4, 5, 6]))


def test_normalize_scan_energy_by_n2_uses_fixed_n2_formula():
    alpha_grid = np.array([1.2, 1.4], dtype=float)
    free_grid = np.array([1, 2], dtype=int)
    total_ion_count_surface = np.array([[7, 7], [11, 11]], dtype=int)
    energy_surface = np.array([[49.0, 98.0], [121.0, 242.0]], dtype=float)
    success_mask = np.ones_like(energy_surface, dtype=bool)
    scan_result = _make_stub_scan_result(
        num_ions_vert=1,
        num_ions_hor=1,
        alpha_grid=alpha_grid,
        num_ions_free_grid=free_grid,
        energy_surface=energy_surface,
        success_mask=success_mask,
        total_ion_count_surface=total_ion_count_surface,
    )

    normalized = normalize_scan_energy_by_n2(scan_result)
    expected = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=float)
    np.testing.assert_allclose(normalized, expected)


def test_polynomial_baseline_fit_supports_degrees_2_3_4_and_skips_failed_points():
    alpha_grid = np.array([1.0, 1.25, 1.5, 1.75, 2.0], dtype=float)
    free_grid = np.array([1, 2, 3, 4], dtype=int)
    alpha_mesh, free_mesh = np.meshgrid(alpha_grid, free_grid, indexing="xy")
    success_mask = np.ones(alpha_mesh.shape, dtype=bool)
    success_mask[1, 2] = False
    total_ion_count_surface = np.full(alpha_mesh.shape, 13, dtype=int)

    for degree in (2, 3, 4):
        normalized_surface = np.zeros_like(alpha_mesh, dtype=float)
        coefficient = 1.0
        for total_degree in range(degree + 1):
            for x_power in range(total_degree, -1, -1):
                y_power = total_degree - x_power
                normalized_surface += coefficient * alpha_mesh**x_power * free_mesh**y_power
                coefficient += 0.05

        energy_surface = normalized_surface * total_ion_count_surface.astype(float) ** 2
        scan_result = _make_stub_scan_result(
            num_ions_vert=1,
            num_ions_hor=2,
            alpha_grid=alpha_grid,
            num_ions_free_grid=free_grid,
            energy_surface=energy_surface,
            success_mask=success_mask,
            total_ion_count_surface=total_ion_count_surface,
        )

        fit = fit_normalized_energy_baseline_polynomial(
            scan_result,
            normalized_surface,
            degree=degree,
        )
        residual = compute_energy_residual_surface(normalized_surface, fit.baseline_surface)

        np.testing.assert_allclose(
            fit.baseline_surface,
            normalized_surface,
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        np.testing.assert_allclose(
            residual,
            np.zeros_like(normalized_surface),
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        assert fit.polynomial_degree == degree
        assert fit.num_points_used == int(np.count_nonzero(success_mask))


def test_compute_median_energy_by_total_ion_count_groups_successful_points_only():
    alpha_grid = np.array([1.2, 1.4, 1.6], dtype=float)
    free_grid = np.array([2, 3], dtype=int)
    energy_surface = np.array([[10.0, 12.0, 14.0], [20.0, 100.0, 24.0]], dtype=float)
    success_mask = np.array([[True, True, True], [True, False, True]], dtype=bool)
    total_ion_count_surface = np.array([[13, 13, 13], [17, 17, 17]], dtype=int)
    scan_result = _make_stub_scan_result(
        num_ions_vert=1,
        num_ions_hor=2,
        alpha_grid=alpha_grid,
        num_ions_free_grid=free_grid,
        energy_surface=energy_surface,
        success_mask=success_mask,
        total_ion_count_surface=total_ion_count_surface,
    )

    median_by_n = compute_median_energy_by_total_ion_count(scan_result)
    np.testing.assert_allclose(median_by_n.total_ion_counts, np.array([13.0, 17.0]))
    np.testing.assert_allclose(median_by_n.median_energies, np.array([12.0, 22.0]))
    np.testing.assert_array_equal(median_by_n.sample_counts, np.array([3, 2]))


def test_fit_energy_n_trend_power_law_and_division_based_scaling():
    alpha_grid = np.array([1.0, 1.5, 2.0], dtype=float)
    free_grid = np.array([1, 2, 3, 4], dtype=int)
    total_ion_count_surface = np.array(
        [[9, 9, 9], [13, 13, 13], [17, 17, 17], [21, 21, 21]],
        dtype=int,
    )
    expected_trend = 5.0 + 0.25 * total_ion_count_surface.astype(float) ** 2.0
    energy_surface = np.array(
        [
            [0.9, 1.0, 1.1],
            [0.9, 1.0, 1.1],
            [0.9, 1.0, 1.1],
            [0.9, 1.0, 1.1],
        ],
        dtype=float,
    ) * expected_trend
    success_mask = np.ones_like(energy_surface, dtype=bool)
    scan_result = _make_stub_scan_result(
        num_ions_vert=1,
        num_ions_hor=1,
        alpha_grid=alpha_grid,
        num_ions_free_grid=free_grid,
        energy_surface=energy_surface,
        success_mask=success_mask,
        total_ion_count_surface=total_ion_count_surface,
    )

    median_by_n = compute_median_energy_by_total_ion_count(scan_result)
    fit = fit_energy_n_trend_power_law(median_by_n)
    scaled = normalize_scan_energy_by_fitted_n_trend(scan_result, fit)

    np.testing.assert_allclose(
        fit.fitted_median_energies,
        median_by_n.median_energies,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    expected_scaled = energy_surface / expected_trend
    np.testing.assert_allclose(scaled, expected_scaled, atol=1.0e-5, rtol=1.0e-5)


def test_best_of_k_warm_start_retries_once_with_generic_seed():
    original_solver = symmetry_scan.solve_symmetry_enforced_harmonic2d
    calls: list[dict[str, object]] = []

    def fake_solver(**kwargs):
        calls.append(kwargs)
        if kwargs["initial_raw_variables"] is not None:
            return SimpleNamespace(
                energy=50.0,
                success=False,
                message="warm start failed",
                reduced_gradient_max_norm=1.0,
            )
        return SimpleNamespace(
            energy=12.0,
            success=True,
            message="generic success",
            reduced_gradient_max_norm=1.0e-6,
        )

    symmetry_scan.solve_symmetry_enforced_harmonic2d = fake_solver
    try:
        result = best_of_k_symmetry_enforced_harmonic2d(
            num_ions_vert=1,
            num_ions_hor=1,
            num_ions_free=2,
            alpha=1.5,
            scale=1.0,
            best_of_k=1,
            rng_seed=11,
            initial_raw_variables=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            initial_seed_metadata={"seed_family": "warm_start_previous_alpha"},
        )
    finally:
        symmetry_scan.solve_symmetry_enforced_harmonic2d = original_solver

    assert len(calls) == 2
    assert calls[0]["initial_raw_variables"] is not None
    assert calls[1]["initial_raw_variables"] is None
    assert result.warm_start_attempted is True
    assert result.fallback_retry_used is True
    assert result.warm_start_succeeded is False
    assert result.final_seed_source == "fallback_generic"
    assert result.attempt_count == 2
    assert result.best_result.success is True
