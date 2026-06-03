from __future__ import annotations

"""High-level alpha/free-ion scan workflow for the symmetry-enforced solver."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from equilibrium_playground.symmetry_enforced_harmonic2d import (
    SymmetryEnforcedHarmonic2DResult,
    solve_symmetry_enforced_harmonic2d,
    symmetry_total_ion_count,
)


@dataclass(frozen=True)
class SymmetryBestOfKResult:
    """Best-of-k wrapper result for one symmetry-enforced scan point."""

    best_result: SymmetryEnforcedHarmonic2DResult | None
    all_results: list[SymmetryEnforcedHarmonic2DResult]
    energies: list[float]
    successful_energies: list[float]
    messages: list[str]
    run_count: int
    attempt_count: int
    warm_start_attempted: bool
    fallback_retry_used: bool
    warm_start_succeeded: bool
    final_seed_source: str | None


@dataclass(frozen=True)
class SymmetryScanPointResult:
    """Stored result for one `(alpha, num_ions_free)` scan point."""

    alpha: float
    num_ions_free: int
    runtime_s: float
    best_of_k_result: SymmetryBestOfKResult
    success: bool
    energy: float
    total_ion_count: int
    reduced_gradient_max_norm: float
    message: str
    warm_start_attempted: bool
    fallback_retry_used: bool
    final_seed_source: str | None

    @property
    def best_result(self) -> SymmetryEnforcedHarmonic2DResult | None:
        return self.best_of_k_result.best_result


@dataclass
class SymmetryFamilyScanResult:
    """2D scan over `alpha` and `num_ions_free` for one symmetry family."""

    num_ions_vert: int
    num_ions_hor: int
    scale: float
    best_of_k: int
    solver_settings: dict[str, Any]
    alpha_grid: np.ndarray
    num_ions_free_grid: np.ndarray
    point_results: np.ndarray
    energy_surface: np.ndarray
    success_mask: np.ndarray
    runtime_surface_s: np.ndarray
    gradient_norm_surface: np.ndarray
    total_ion_count_surface: np.ndarray
    message_surface: np.ndarray

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return meshgrids with `x=alpha`, `y=num_ions_free`."""

        return np.meshgrid(self.alpha_grid, self.num_ions_free_grid, indexing="xy")

    @property
    def shape(self) -> tuple[int, int]:
        return self.energy_surface.shape


@dataclass(frozen=True)
class PolynomialBaselineFitResult:
    """2D polynomial baseline fit for a scaled energy surface on the scan grid."""

    polynomial_degree: int
    coefficients: np.ndarray
    baseline_surface: np.ndarray
    alpha_center: float
    alpha_scale: float
    num_ions_free_center: float
    num_ions_free_scale: float
    num_points_used: int


Degree3PolynomialBaselineFit = PolynomialBaselineFitResult


@dataclass(frozen=True)
class MedianEnergyByTotalIonCount:
    """Median raw energy by total ion count over successful alpha points."""

    total_ion_counts: np.ndarray
    median_energies: np.ndarray
    sample_counts: np.ndarray


@dataclass(frozen=True)
class EnergyNTrendPowerLawFit:
    """Power-law N-trend fit with model `c0 + c1 * N^p`."""

    c0: float
    c1: float
    p: float
    total_ion_counts: np.ndarray
    median_energies: np.ndarray
    fitted_median_energies: np.ndarray
    sample_counts: np.ndarray
    num_points_used: int


def build_alpha_grid(alpha_min: float, alpha_max: float, num_alphas: int) -> np.ndarray:
    """Return the inclusive linearly spaced alpha grid."""

    alpha_start = float(alpha_min)
    alpha_stop = float(alpha_max)
    alpha_count = int(num_alphas)
    if alpha_count <= 0:
        raise ValueError("num_alphas must be positive")
    if alpha_start <= 0.0 or alpha_stop <= 0.0:
        raise ValueError("alpha_min and alpha_max must be positive")
    if alpha_stop < alpha_start:
        raise ValueError("alpha_max must be greater than or equal to alpha_min")
    return np.linspace(alpha_start, alpha_stop, alpha_count, dtype=float)


def build_num_ions_free_grid(
    num_ions_free_min: int,
    num_ions_free_max: int,
) -> np.ndarray:
    """Return the inclusive integer `num_ions_free` grid."""

    free_min = int(num_ions_free_min)
    free_max = int(num_ions_free_max)
    if free_min < 0:
        raise ValueError("num_ions_free_min must be nonnegative")
    if free_max < free_min:
        raise ValueError("num_ions_free_max must be greater than or equal to num_ions_free_min")
    return np.arange(free_min, free_max + 1, dtype=int)


def _is_successful_result(result: SymmetryEnforcedHarmonic2DResult | None) -> bool:
    return (
        result is not None
        and bool(result.success)
        and np.isfinite(float(result.energy))
    )


def _warm_start_seed_metadata(
    rng_seed: int | None,
) -> dict[str, Any]:
    return {
        "seed_family": "warm_start_previous_alpha",
        "rng_seed": rng_seed,
    }


def best_of_k_symmetry_enforced_harmonic2d(
    *,
    num_ions_vert: int,
    num_ions_hor: int,
    num_ions_free: int,
    alpha: float,
    scale: float,
    best_of_k: int = 1,
    seed_scale: float = 1.0,
    vertical_seed_scale_factor: float = 1.0,
    seed_jitter: float = 0.0,
    rng_seed: int | None = None,
    initial_raw_variables: np.ndarray | None = None,
    initial_seed_metadata: dict[str, Any] | None = None,
    basinhopping_niter: int = 60,
    basin_temperature: float = 0.35,
    basin_stepsize: float = 0.3,
    bfgs_gtol: float = 1.0e-10,
    bfgs_maxiter: int = 1000,
    final_gtol: float = 1.0e-11,
    final_ftol: float = 1.0e-14,
    final_maxiter: int = 4000,
) -> SymmetryBestOfKResult:
    """Run the symmetry-enforced solver up to `k` times and keep the best result."""

    run_count = int(best_of_k)
    if run_count <= 0:
        raise ValueError("best_of_k must be positive")

    results: list[SymmetryEnforcedHarmonic2DResult] = []
    messages: list[str] = []
    result_sources: dict[int, str] = {}
    attempt_count = 0
    warm_start_attempted = initial_raw_variables is not None
    fallback_retry_used = False
    warm_start_succeeded = False

    def run_attempt(
        seed_source: str,
        *,
        start_raw_variables: np.ndarray | None,
        start_seed_metadata: dict[str, Any] | None,
    ) -> SymmetryEnforcedHarmonic2DResult | None:
        nonlocal attempt_count

        run_rng_seed = None if rng_seed is None else int(rng_seed) + attempt_count
        attempt_count += 1
        try:
            result = solve_symmetry_enforced_harmonic2d(
                num_ions_vert=num_ions_vert,
                num_ions_hor=num_ions_hor,
                num_ions_free=num_ions_free,
                alpha=float(alpha),
                scale=float(scale),
                seed_scale=seed_scale,
                vertical_seed_scale_factor=vertical_seed_scale_factor,
                seed_jitter=seed_jitter,
                rng_seed=run_rng_seed,
                initial_raw_variables=start_raw_variables,
                initial_seed_metadata=start_seed_metadata,
                basinhopping_niter=basinhopping_niter,
                basin_temperature=basin_temperature,
                basin_stepsize=basin_stepsize,
                bfgs_gtol=bfgs_gtol,
                bfgs_maxiter=bfgs_maxiter,
                final_gtol=final_gtol,
                final_ftol=final_ftol,
                final_maxiter=final_maxiter,
            )
            results.append(result)
            result_sources[id(result)] = seed_source
            messages.append(str(result.message))
            return result
        except Exception as exc:
            messages.append(f"{seed_source}: exception: {exc}")
            return None

    primary_runs_remaining = run_count
    if warm_start_attempted:
        warm_result = run_attempt(
            "warm_start",
            start_raw_variables=np.asarray(initial_raw_variables, dtype=float).reshape(-1),
            start_seed_metadata=initial_seed_metadata,
        )
        primary_runs_remaining -= 1
        if _is_successful_result(warm_result):
            warm_start_succeeded = True
        else:
            fallback_retry_used = True
            run_attempt(
                "fallback_generic",
                start_raw_variables=None,
                start_seed_metadata=None,
            )

    for _ in range(max(primary_runs_remaining, 0)):
        run_attempt(
            "generic",
            start_raw_variables=None,
            start_seed_metadata=None,
        )

    successful_results = [result for result in results if _is_successful_result(result)]
    ranked_results = successful_results if successful_results else results
    best_result = min(ranked_results, key=lambda result: float(result.energy)) if ranked_results else None

    return SymmetryBestOfKResult(
        best_result=best_result,
        all_results=results,
        energies=[float(result.energy) for result in results],
        successful_energies=[float(result.energy) for result in successful_results],
        messages=messages,
        run_count=run_count,
        attempt_count=int(attempt_count),
        warm_start_attempted=bool(warm_start_attempted),
        fallback_retry_used=bool(fallback_retry_used),
        warm_start_succeeded=bool(warm_start_succeeded),
        final_seed_source=None if best_result is None else result_sources.get(id(best_result)),
    )


def scan_symmetry_family_over_alpha_and_free(
    *,
    num_ions_vert: int,
    num_ions_hor: int,
    scale: float,
    alpha_min: float,
    alpha_max: float,
    num_alphas: int,
    num_ions_free_min: int,
    num_ions_free_max: int,
    best_of_k: int = 1,
    seed_scale: float = 1.0,
    vertical_seed_scale_factor: float = 1.0,
    seed_jitter: float = 0.0,
    rng_seed: int | None = None,
    basinhopping_niter: int = 60,
    basin_temperature: float = 0.35,
    basin_stepsize: float = 0.3,
    bfgs_gtol: float = 1.0e-10,
    bfgs_maxiter: int = 1000,
    final_gtol: float = 1.0e-11,
    final_ftol: float = 1.0e-14,
    final_maxiter: int = 4000,
) -> SymmetryFamilyScanResult:
    """Scan a rectangular `(alpha, num_ions_free)` grid for one symmetry family."""

    alpha_grid = build_alpha_grid(alpha_min, alpha_max, num_alphas)
    num_ions_free_grid = build_num_ions_free_grid(num_ions_free_min, num_ions_free_max)
    shape = (num_ions_free_grid.size, alpha_grid.size)

    point_results = np.empty(shape, dtype=object)
    energy_surface = np.full(shape, np.nan, dtype=float)
    success_mask = np.zeros(shape, dtype=bool)
    runtime_surface_s = np.full(shape, np.nan, dtype=float)
    gradient_norm_surface = np.full(shape, np.nan, dtype=float)
    total_ion_count_surface = np.zeros(shape, dtype=int)
    message_surface = np.empty(shape, dtype=object)

    solver_settings = {
        "seed_scale": float(seed_scale),
        "vertical_seed_scale_factor": float(vertical_seed_scale_factor),
        "seed_jitter": float(seed_jitter),
        "rng_seed": rng_seed,
        "basinhopping_niter": int(basinhopping_niter),
        "basin_temperature": float(basin_temperature),
        "basin_stepsize": float(basin_stepsize),
        "bfgs_gtol": float(bfgs_gtol),
        "bfgs_maxiter": int(bfgs_maxiter),
        "final_gtol": float(final_gtol),
        "final_ftol": float(final_ftol),
        "final_maxiter": int(final_maxiter),
    }

    for row_index, num_ions_free in enumerate(num_ions_free_grid):
        total_ion_count = symmetry_total_ion_count(
            num_ions_vert,
            num_ions_hor,
            int(num_ions_free),
        )
        previous_successful_result: SymmetryEnforcedHarmonic2DResult | None = None
        for col_index, alpha in enumerate(alpha_grid):
            warm_start_raw = None
            warm_start_metadata = None
            if previous_successful_result is not None:
                warm_start_raw = previous_successful_result.optimizer_variables_raw.copy()
                warm_start_metadata = _warm_start_seed_metadata(rng_seed)

            print(
                "scan point: "
                f"alpha={float(alpha):.6g}, "
                f"num_ions_free={int(num_ions_free)}, "
                f"initial={'warm-start' if warm_start_raw is not None else 'generic'}"
            )
            t_start = perf_counter()
            best_bundle = best_of_k_symmetry_enforced_harmonic2d(
                num_ions_vert=num_ions_vert,
                num_ions_hor=num_ions_hor,
                num_ions_free=int(num_ions_free),
                alpha=float(alpha),
                scale=float(scale),
                best_of_k=best_of_k,
                seed_scale=seed_scale,
                vertical_seed_scale_factor=vertical_seed_scale_factor,
                seed_jitter=seed_jitter,
                rng_seed=rng_seed,
                initial_raw_variables=warm_start_raw,
                initial_seed_metadata=warm_start_metadata,
                basinhopping_niter=basinhopping_niter,
                basin_temperature=basin_temperature,
                basin_stepsize=basin_stepsize,
                bfgs_gtol=bfgs_gtol,
                bfgs_maxiter=bfgs_maxiter,
                final_gtol=final_gtol,
                final_ftol=final_ftol,
                final_maxiter=final_maxiter,
            )
            runtime_s = perf_counter() - t_start

            best_result = best_bundle.best_result
            if best_result is None:
                success = False
                energy = np.nan
                gradient_norm = np.nan
                message = "; ".join(best_bundle.messages) if best_bundle.messages else "no result"
                previous_successful_result = None
                print(
                    "warning: scan point produced no solver result: "
                    f"alpha={float(alpha):.6g}, "
                    f"num_ions_free={int(num_ions_free)}"
                )
            else:
                success = _is_successful_result(best_result)
                energy = float(best_result.energy)
                gradient_norm = float(best_result.reduced_gradient_max_norm)
                message = str(best_result.message)
                if success:
                    previous_successful_result = best_result
                    print(
                        "completed: "
                        f"energy={energy:.10g}, "
                        f"runtime={runtime_s:.3f} s, "
                        f"source={best_bundle.final_seed_source}"
                    )
                else:
                    previous_successful_result = None
                    print(
                        "warning: scan point failed: "
                        f"alpha={float(alpha):.6g}, "
                        f"num_ions_free={int(num_ions_free)}, "
                        f"message={message}"
                    )

            point_result = SymmetryScanPointResult(
                alpha=float(alpha),
                num_ions_free=int(num_ions_free),
                runtime_s=float(runtime_s),
                best_of_k_result=best_bundle,
                success=bool(success),
                energy=float(energy),
                total_ion_count=int(total_ion_count),
                reduced_gradient_max_norm=float(gradient_norm),
                message=message,
                warm_start_attempted=bool(best_bundle.warm_start_attempted),
                fallback_retry_used=bool(best_bundle.fallback_retry_used),
                final_seed_source=best_bundle.final_seed_source,
            )

            point_results[row_index, col_index] = point_result
            energy_surface[row_index, col_index] = energy
            success_mask[row_index, col_index] = bool(success)
            runtime_surface_s[row_index, col_index] = float(runtime_s)
            gradient_norm_surface[row_index, col_index] = float(gradient_norm)
            total_ion_count_surface[row_index, col_index] = int(total_ion_count)
            message_surface[row_index, col_index] = message

    return SymmetryFamilyScanResult(
        num_ions_vert=int(num_ions_vert),
        num_ions_hor=int(num_ions_hor),
        scale=float(scale),
        best_of_k=int(best_of_k),
        solver_settings=solver_settings,
        alpha_grid=alpha_grid,
        num_ions_free_grid=num_ions_free_grid,
        point_results=point_results,
        energy_surface=energy_surface,
        success_mask=success_mask,
        runtime_surface_s=runtime_surface_s,
        gradient_norm_surface=gradient_norm_surface,
        total_ion_count_surface=total_ion_count_surface,
        message_surface=message_surface,
    )


def normalize_scan_energy_by_n2(scan_result: SymmetryFamilyScanResult) -> np.ndarray:
    """Return the fixed `E / N^2` normalized energy surface."""

    total_ion_count_sq = np.square(scan_result.total_ion_count_surface.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = scan_result.energy_surface / total_ion_count_sq
    return normalized


def compute_median_energy_by_total_ion_count(
    scan_result: SymmetryFamilyScanResult,
) -> MedianEnergyByTotalIonCount:
    """Compute median raw energy over alpha for each total ion count `N`.

    Only successful finite scan points are included in the aggregation.
    """

    valid_mask = scan_result.success_mask & np.isfinite(scan_result.energy_surface)
    if not np.any(valid_mask):
        raise ValueError("no successful finite scan points are available for median aggregation")

    total_ion_counts = scan_result.total_ion_count_surface[valid_mask].astype(int)
    energies = scan_result.energy_surface[valid_mask].astype(float)
    unique_total_ion_counts = np.unique(total_ion_counts)

    median_energies = np.empty(unique_total_ion_counts.size, dtype=float)
    sample_counts = np.empty(unique_total_ion_counts.size, dtype=int)
    for index, total_ion_count in enumerate(unique_total_ion_counts):
        group_mask = total_ion_counts == total_ion_count
        median_energies[index] = float(np.median(energies[group_mask]))
        sample_counts[index] = int(np.count_nonzero(group_mask))

    return MedianEnergyByTotalIonCount(
        total_ion_counts=unique_total_ion_counts.astype(float),
        median_energies=median_energies,
        sample_counts=sample_counts,
    )


def _energy_n_trend_model(total_ion_count: np.ndarray, c0: float, c1: float, p: float) -> np.ndarray:
    n = np.asarray(total_ion_count, dtype=float)
    return c0 + c1 * np.power(n, p)


def fit_energy_n_trend_power_law(
    median_energy_by_total_ion_count: MedianEnergyByTotalIonCount,
) -> EnergyNTrendPowerLawFit:
    """Fit the 1D N-trend model `E_trend(N) = c0 + c1 * N^p`.

    The fit is performed on the median raw energy over alpha at each unique
    total ion count using nonlinear least squares.
    """

    total_ion_counts = np.asarray(median_energy_by_total_ion_count.total_ion_counts, dtype=float)
    median_energies = np.asarray(median_energy_by_total_ion_count.median_energies, dtype=float)
    sample_counts = np.asarray(median_energy_by_total_ion_count.sample_counts, dtype=int)

    valid_mask = np.isfinite(total_ion_counts) & np.isfinite(median_energies)
    total_ion_counts = total_ion_counts[valid_mask]
    median_energies = median_energies[valid_mask]
    sample_counts = sample_counts[valid_mask]

    if total_ion_counts.size < 4:
        raise ValueError(
            "at least four unique total ion counts are required to fit "
            "E_trend(N) = c0 + c1 * N^p"
        )

    best_parameters: np.ndarray | None = None
    best_residual = np.inf
    for p_guess in (1.0, 2.0, 3.0):
        n_first = float(total_ion_counts[0])
        n_last = float(total_ion_counts[-1])
        y_first = float(median_energies[0])
        y_last = float(median_energies[-1])
        denom = max(n_last**p_guess - n_first**p_guess, 1.0e-12)
        c1_guess = (y_last - y_first) / denom
        if not np.isfinite(c1_guess) or c1_guess <= 0.0:
            c1_guess = max(float(np.median(median_energies)) / max(n_last**p_guess, 1.0), 1.0e-12)
        c0_guess = float(np.median(median_energies - c1_guess * np.power(total_ion_counts, p_guess)))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                parameters, _ = curve_fit(
                    _energy_n_trend_model,
                    total_ion_counts,
                    median_energies,
                    p0=(c0_guess, c1_guess, p_guess),
                    bounds=([-np.inf, 0.0, 0.1], [np.inf, np.inf, 6.0]),
                    maxfev=20000,
                )
        except Exception:
            continue

        fitted = _energy_n_trend_model(total_ion_counts, *parameters)
        if np.any(~np.isfinite(fitted)) or np.any(fitted <= 0.0):
            continue

        residual = float(np.sum((fitted - median_energies) ** 2))
        if residual < best_residual:
            best_parameters = parameters
            best_residual = residual

    if best_parameters is None:
        raise RuntimeError("failed to fit a sensible power-law N-trend")

    fitted_median_energies = _energy_n_trend_model(total_ion_counts, *best_parameters)
    return EnergyNTrendPowerLawFit(
        c0=float(best_parameters[0]),
        c1=float(best_parameters[1]),
        p=float(best_parameters[2]),
        total_ion_counts=total_ion_counts,
        median_energies=median_energies,
        fitted_median_energies=fitted_median_energies,
        sample_counts=sample_counts,
        num_points_used=int(total_ion_counts.size),
    )


def evaluate_energy_n_trend_on_scan(
    scan_result: SymmetryFamilyScanResult,
    n_trend_fit: EnergyNTrendPowerLawFit,
) -> np.ndarray:
    """Evaluate the fitted N-trend on the full scan grid."""

    trend_surface = _energy_n_trend_model(
        scan_result.total_ion_count_surface.astype(float),
        n_trend_fit.c0,
        n_trend_fit.c1,
        n_trend_fit.p,
    )
    if np.any(~np.isfinite(trend_surface)) or np.any(trend_surface <= 0.0):
        raise ValueError("the fitted N-trend is not positive and finite on the scan grid")
    return trend_surface


def normalize_scan_energy_by_fitted_n_trend(
    scan_result: SymmetryFamilyScanResult,
    n_trend_fit: EnergyNTrendPowerLawFit,
) -> np.ndarray:
    """Return the division-based first-stage scaled energy `E / E_trend(N)`."""

    trend_surface = evaluate_energy_n_trend_on_scan(scan_result, n_trend_fit)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = scan_result.energy_surface / trend_surface
    scaled[~np.isfinite(scan_result.energy_surface)] = np.nan
    return scaled


def _scaled_coordinates(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    center = 0.5 * (float(flat.min()) + float(flat.max()))
    scale = 0.5 * (float(flat.max()) - float(flat.min()))
    if scale <= 0.0:
        scale = 1.0
    return (flat - center) / scale, center, scale


def _polynomial_design_matrix(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    """Build the total-degree polynomial design matrix in two variables."""

    polynomial_degree = int(degree)
    if polynomial_degree < 0:
        raise ValueError("degree must be nonnegative")

    x_arr = np.asarray(x_scaled, dtype=float).reshape(-1)
    y_arr = np.asarray(y_scaled, dtype=float).reshape(-1)
    columns = []
    for total_degree in range(polynomial_degree + 1):
        for x_power in range(total_degree, -1, -1):
            y_power = total_degree - x_power
            columns.append(np.power(x_arr, x_power) * np.power(y_arr, y_power))
    return np.column_stack(columns)


def fit_normalized_energy_baseline_polynomial(
    scan_result: SymmetryFamilyScanResult,
    normalized_energy_surface: np.ndarray | None = None,
    *,
    degree: int = 3,
) -> PolynomialBaselineFitResult:
    """Fit a total-degree polynomial baseline over successful points only.

    Despite the historical function name, this helper accepts any scaled energy
    surface defined on the scan grid, including the old `E / N^2` surface and
    the new `E / E_trend(N)` surface.
    """

    polynomial_degree = int(degree)
    if polynomial_degree not in (2, 3, 4):
        raise ValueError("baseline polynomial degree must be one of 2, 3, or 4")

    normalized = (
        normalize_scan_energy_by_n2(scan_result)
        if normalized_energy_surface is None
        else np.asarray(normalized_energy_surface, dtype=float)
    )
    if normalized.shape != scan_result.shape:
        raise ValueError("normalized_energy_surface has the wrong shape")

    alpha_mesh, free_mesh = scan_result.meshgrid()
    fit_mask = scan_result.success_mask & np.isfinite(normalized)
    if not np.any(fit_mask):
        raise ValueError("no successful finite scan points are available for the baseline fit")

    alpha_scaled_all, alpha_center, alpha_scale = _scaled_coordinates(alpha_mesh)
    free_scaled_all, free_center, free_scale = _scaled_coordinates(free_mesh)
    alpha_scaled_all = alpha_scaled_all.reshape(scan_result.shape)
    free_scaled_all = free_scaled_all.reshape(scan_result.shape)

    design = _polynomial_design_matrix(
        alpha_scaled_all[fit_mask],
        free_scaled_all[fit_mask],
        degree=polynomial_degree,
    )
    values = normalized[fit_mask]
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)

    full_design = _polynomial_design_matrix(
        alpha_scaled_all.reshape(-1),
        free_scaled_all.reshape(-1),
        degree=polynomial_degree,
    )
    baseline_surface = (full_design @ coefficients).reshape(scan_result.shape)

    return PolynomialBaselineFitResult(
        polynomial_degree=polynomial_degree,
        coefficients=coefficients,
        baseline_surface=baseline_surface,
        alpha_center=float(alpha_center),
        alpha_scale=float(alpha_scale),
        num_ions_free_center=float(free_center),
        num_ions_free_scale=float(free_scale),
        num_points_used=int(np.count_nonzero(fit_mask)),
    )


def fit_normalized_energy_baseline_degree3(
    scan_result: SymmetryFamilyScanResult,
    normalized_energy_surface: np.ndarray | None = None,
) -> PolynomialBaselineFitResult:
    """Compatibility wrapper for the legacy degree-3 baseline fit path."""

    return fit_normalized_energy_baseline_polynomial(
        scan_result,
        normalized_energy_surface,
        degree=3,
    )


def compute_energy_residual_surface(
    normalized_energy_surface: np.ndarray,
    baseline_surface: np.ndarray,
) -> np.ndarray:
    """Return `normalized_energy - fitted_baseline` with NaNs preserved."""

    normalized = np.asarray(normalized_energy_surface, dtype=float)
    baseline = np.asarray(baseline_surface, dtype=float)
    if normalized.shape != baseline.shape:
        raise ValueError("normalized_energy_surface and baseline_surface must have the same shape")

    residual = normalized - baseline
    residual[~np.isfinite(normalized)] = np.nan
    return residual


def _plot_scan_surface(
    scan_result: SymmetryFamilyScanResult,
    surface: np.ndarray,
    *,
    title: str,
    zlabel: str,
    cmap: str,
):
    values = np.asarray(surface, dtype=float)
    if values.shape != scan_result.shape:
        raise ValueError("surface has the wrong shape")

    alpha_mesh, free_mesh = scan_result.meshgrid()
    fig = plt.figure(figsize=(8.2, 6.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    surface_plot = ax.plot_surface(
        alpha_mesh,
        free_mesh,
        values,
        cmap=cmap,
        linewidth=0.0,
        antialiased=True,
        alpha=0.82,
    )

    successful_mask = scan_result.success_mask & np.isfinite(values)
    failed_mask = (~scan_result.success_mask) & np.isfinite(values)
    if np.any(successful_mask):
        ax.scatter(
            alpha_mesh[successful_mask],
            free_mesh[successful_mask],
            values[successful_mask],
            c="black",
            s=24,
            depthshade=False,
            label="successful points",
        )
    if np.any(failed_mask):
        ax.scatter(
            alpha_mesh[failed_mask],
            free_mesh[failed_mask],
            values[failed_mask],
            c="#d62728",
            marker="x",
            s=36,
            depthshade=False,
            label="failed points",
        )

    ax.set_xlabel("alpha")
    ax.set_ylabel("num_ions_free")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surface_plot, ax=ax, shrink=0.72, pad=0.08)
    if np.any(successful_mask) or np.any(failed_mask):
        ax.legend(loc="best")
    return fig


def plot_scan_raw_energy_surface(scan_result: SymmetryFamilyScanResult):
    """Plot the raw total energy surface with measured-point overlays."""

    return _plot_scan_surface(
        scan_result,
        scan_result.energy_surface,
        title="Raw total energy surface",
        zlabel="total energy",
        cmap="viridis",
    )


def plot_scan_normalized_energy_surface(
    scan_result: SymmetryFamilyScanResult,
    normalized_energy_surface: np.ndarray | None = None,
):
    """Plot the normalized `E / N^2` energy surface with measured-point overlays."""

    normalized = (
        normalize_scan_energy_by_n2(scan_result)
        if normalized_energy_surface is None
        else np.asarray(normalized_energy_surface, dtype=float)
    )
    return _plot_scan_surface(
        scan_result,
        normalized,
        title="Normalized energy surface",
        zlabel="E / N^2",
        cmap="plasma",
    )


def plot_scan_residual_surface(
    scan_result: SymmetryFamilyScanResult,
    residual_surface: np.ndarray,
    *,
    baseline_degree: int | None = None,
):
    """Plot the detrended residual surface with measured-point overlays."""

    if baseline_degree is None:
        title = "Residual surface after polynomial baseline subtraction"
        zlabel = "normalized residual"
    else:
        degree_value = int(baseline_degree)
        title = f"Residual surface after degree-{degree_value} baseline subtraction"
        zlabel = f"residual after degree-{degree_value} baseline"

    return _plot_scan_surface(
        scan_result,
        residual_surface,
        title=title,
        zlabel=zlabel,
        cmap="coolwarm",
    )
