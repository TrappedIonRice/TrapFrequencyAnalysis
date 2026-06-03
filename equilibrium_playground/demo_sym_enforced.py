from __future__ import annotations

"""Standalone demo for the symmetry-enforced harmonic 2D playground solver."""

import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter

from equilibrium_playground.symmetry_enforced_harmonic2d import (
    solve_symmetry_enforced_harmonic2d,
)
from equilibrium_playground.symmetry_scan import (
    compute_median_energy_by_total_ion_count,
    compute_energy_residual_surface,
    fit_energy_n_trend_power_law,
    fit_normalized_energy_baseline_polynomial,
    normalize_scan_energy_by_n2,
    normalize_scan_energy_by_fitted_n_trend,
    plot_scan_normalized_energy_surface,
    plot_scan_raw_energy_surface,
    plot_scan_residual_surface,
    scan_symmetry_family_over_alpha_and_free,
)
from equilibrium_playground.unrestricted_harmonic2d import (
    compare_symmetry_seed_to_unrestricted_refinement,
)
from equilibrium_playground.viewer import build_result_figure, print_result_summary


_RESIDUAL_BASELINE_DEGREES = (2, 3, 4)


def _compute_residual_analysis_by_degree(
    scan_result,
    scaled_energy_surface: np.ndarray,
    *,
    baseline_degrees: tuple[int, ...] = _RESIDUAL_BASELINE_DEGREES,
) -> dict[int, dict[str, object]]:
    analysis: dict[int, dict[str, object]] = {}
    for degree in baseline_degrees:
        baseline_fit = fit_normalized_energy_baseline_polynomial(
            scan_result,
            scaled_energy_surface,
            degree=int(degree),
        )
        residual_surface = compute_energy_residual_surface(
            scaled_energy_surface,
            baseline_fit.baseline_surface,
        )
        analysis[int(degree)] = {
            "baseline_fit": baseline_fit,
            "residual_surface": residual_surface,
        }
    return analysis


def _print_residual_summary_by_degree(
    scan_result,
    scaled_energy_surface: np.ndarray,
    residual_analysis_by_degree: dict[int, dict[str, object]],
) -> None:
    for degree in sorted(residual_analysis_by_degree):
        baseline_fit = residual_analysis_by_degree[degree]["baseline_fit"]
        residual_surface = residual_analysis_by_degree[degree]["residual_surface"]
        print(f"baseline degree {degree} fit points used: {baseline_fit.num_points_used}")

        residual_on_success = np.where(scan_result.success_mask, residual_surface, np.nan)
        if np.any(np.isfinite(residual_on_success)):
            min_index = np.unravel_index(np.nanargmin(residual_on_success), residual_on_success.shape)
            min_row, min_col = int(min_index[0]), int(min_index[1])
            print(
                f"most negative degree-{degree} residual: "
                f"alpha={float(scan_result.alpha_grid[min_col]):.6g}, "
                f"num_ions_free={int(scan_result.num_ions_free_grid[min_row])}, "
                f"residual={float(residual_surface[min_row, min_col]):.6g}, "
                f"scaled={float(scaled_energy_surface[min_row, min_col]):.6g}, "
                f"energy={float(scan_result.energy_surface[min_row, min_col]):.6g}"
            )


def _plot_symmetry_vs_unrestricted_comparison(comparison_result) -> None:
    symmetry_points = np.asarray(
        comparison_result.symmetry_result.positions_2d,
        dtype=float,
    ).reshape(-1, 2)
    unrestricted_points = np.asarray(
        comparison_result.unrestricted_result.positions_2d,
        dtype=float,
    ).reshape(-1, 2)

    all_points = np.vstack([symmetry_points, unrestricted_points])
    x_pad = 0.05 * max(float(np.ptp(all_points[:, 0])), 1.0)
    z_pad = 0.05 * max(float(np.ptp(all_points[:, 1])), 1.0)
    x_limits = (
        float(np.min(all_points[:, 0]) - x_pad),
        float(np.max(all_points[:, 0]) + x_pad),
    )
    z_limits = (
        float(np.min(all_points[:, 1]) - z_pad),
        float(np.max(all_points[:, 1]) + z_pad),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    axes[0].scatter(
        symmetry_points[:, 0],
        symmetry_points[:, 1],
        s=55,
        c="#1f77b4",
        edgecolors="black",
        linewidths=0.5,
    )
    axes[0].set_title(
        "Symmetry-enforced seed\n"
        f"E = {float(comparison_result.symmetry_energy):.8g}"
    )

    axes[1].scatter(
        unrestricted_points[:, 0],
        unrestricted_points[:, 1],
        s=55,
        c="#d62728",
        edgecolors="black",
        linewidths=0.5,
    )
    axes[1].set_title(
        "Unrestricted refinement\n"
        f"E = {float(comparison_result.unrestricted_energy):.8g}"
    )

    for ax in axes:
        ax.set_xlabel("x (dimensionless)")
        ax.set_ylabel("z (dimensionless)")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*z_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Symmetry vs unrestricted harmonic 2D refinement\n"
        f"delta E = {float(comparison_result.delta_energy):.6g}, "
        f"sorted RMS move = {float(comparison_result.rms_displacement_sorted):.6g}"
    )


def main() -> None:
    result = solve_symmetry_enforced_harmonic2d(
        num_ions_vert=1,
        num_ions_hor=6,
        num_ions_free=19,
        alpha=1.9,
        scale=1.0,
        seed_scale=1.0,
        vertical_seed_scale_factor=3.0,
        seed_jitter=0.0,
        rng_seed=7,
        basinhopping_niter=40,
    )

    print_result_summary(result)
    print()
    print("reduced variables:")
    for key, value in result.reduced_variables.items():
        print(f"{key}:")
        print(value)

    fig = build_result_figure(
        result,
        title="Symmetry-enforced harmonic 2D equilibrium",
    )
    if fig is not None:
        plt.show()


def scan_num_ions_free_runtime(
    *,
    n_min: int = 1,
    n_max: int = 40,
    num_ions_vert: int = 1,
    num_ions_hor: int = 3,
    alpha: float = 1.9,
    scale: float = 1.0,
    seed_scale: float = 1.0,
    vertical_seed_scale_factor: float = 1.0,
    seed_jitter: float = 0.0,
    rng_seed: int | None = 7,
    basinhopping_niter: int = 40,
) -> tuple[list[int], list[float]]:
    """Measure solve time while scanning `num_ions_free` over a fixed range."""

    n_start = int(n_min)
    n_stop = int(n_max)
    if n_start <= 0:
        raise ValueError("n_min must be positive")
    if n_stop < n_start:
        raise ValueError("n_max must be greater than or equal to n_min")

    free_ion_counts: list[int] = []
    solve_times_s: list[float] = []

    for num_ions_free in range(n_start, n_stop + 1):
        t_start = perf_counter()
        solve_symmetry_enforced_harmonic2d(
            num_ions_vert=num_ions_vert,
            num_ions_hor=num_ions_hor,
            num_ions_free=num_ions_free,
            alpha=alpha,
            scale=scale,
            seed_scale=seed_scale,
            vertical_seed_scale_factor=vertical_seed_scale_factor,
            seed_jitter=seed_jitter,
            rng_seed=rng_seed,
            basinhopping_niter=basinhopping_niter,
        )
        print(f"num_ions_free={num_ions_free} solve time: {perf_counter() - t_start:.3f} s")
        solve_times_s.append(perf_counter() - t_start)
        free_ion_counts.append(num_ions_free)

    return free_ion_counts, solve_times_s


def main_2() -> None:
    free_ion_counts, solve_times_s = scan_num_ions_free_runtime()
    x_data = np.asarray(free_ion_counts, dtype=float)
    y_data = np.asarray(solve_times_s, dtype=float)
    x_dense = np.linspace(x_data.min(), x_data.max(), 600)

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(
        free_ion_counts,
        solve_times_s,
        "o",
        color="#1f77b4",
        markersize=4.0,
        label="measured runtime",
    )
    if x_data.size >= 3:
        poly2 = np.poly1d(np.polyfit(x_data, y_data, deg=2))
        ax.plot(x_dense, poly2(x_dense), "-", color="#d62728", linewidth=1.4, label="poly deg 2 fit")
    if x_data.size >= 4:
        poly3 = np.poly1d(np.polyfit(x_data, y_data, deg=3))
        ax.plot(x_dense, poly3(x_dense), "-", color="#2ca02c", linewidth=1.4, label="poly deg 3 fit")
    if x_data.size >= 5:
        poly4 = np.poly1d(np.polyfit(x_data, y_data, deg=4))
        ax.plot(x_dense, poly4(x_dense), "-", color="#ff7f0e", linewidth=1.4, label="poly deg 4 fit")
    if x_data.size >= 2:
        log_y = np.log(np.maximum(y_data, 1.0e-12))
        exp_slope, exp_intercept = np.polyfit(x_data, log_y, deg=1)
        exp_fit = np.exp(exp_intercept + exp_slope * x_dense)
        ax.plot(x_dense, exp_fit, "--", color="#9467bd", linewidth=1.6, label="exponential fit")
    ax.set_xlabel("num_ions_free")
    ax.set_ylabel("solve time (s)")
    ax.set_title("SymmetryEnforced_Harmonic2D runtime vs free first-quadrant ions")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.show()


def main_3() -> None:
    scan_result = scan_symmetry_family_over_alpha_and_free(
        num_ions_vert=1,
        num_ions_hor=6,
        scale=1.0,
        alpha_min=1.9,
        alpha_max=2.1,
        num_alphas=5,
        num_ions_free_min=12,
        num_ions_free_max=28,
        best_of_k=1,
        seed_scale=1.0,
        vertical_seed_scale_factor=2.0,
        seed_jitter=0.0,
        rng_seed=7,
        basinhopping_niter=40,
    )

    normalized_energy_n2 = normalize_scan_energy_by_n2(scan_result)
    median_energy_by_n = compute_median_energy_by_total_ion_count(scan_result)
    n_trend_fit = fit_energy_n_trend_power_law(median_energy_by_n)
    scaled_energy_by_n_trend = normalize_scan_energy_by_fitted_n_trend(scan_result, n_trend_fit)
    residual_analysis_by_degree = _compute_residual_analysis_by_degree(
        scan_result,
        scaled_energy_by_n_trend,
    )

    successful_count = int(np.count_nonzero(scan_result.success_mask))
    total_count = int(scan_result.success_mask.size)
    mean_runtime = float(np.nanmean(scan_result.runtime_surface_s))

    print("SymmetryEnforced_Harmonic2D alpha/free scan")
    print(
        "fixed inputs: "
        f"num_ions_vert={scan_result.num_ions_vert}, "
        f"num_ions_hor={scan_result.num_ions_hor}, "
        f"scale={scan_result.scale}, "
        f"best_of_k={scan_result.best_of_k}"
    )
    print(
        "alpha grid: "
        f"{float(scan_result.alpha_grid[0]):.6g} .. "
        f"{float(scan_result.alpha_grid[-1]):.6g} "
        f"({len(scan_result.alpha_grid)} points)"
    )
    print(
        "num_ions_free grid: "
        f"{int(scan_result.num_ions_free_grid[0])} .. "
        f"{int(scan_result.num_ions_free_grid[-1])}"
    )
    print(f"successful points: {successful_count}/{total_count}")
    print(f"mean runtime per point: {mean_runtime:.3f} s")
    print(
        "N-trend fit: "
        f"c0={n_trend_fit.c0:.6g}, "
        f"c1={n_trend_fit.c1:.6g}, "
        f"p={n_trend_fit.p:.6g}"
    )
    print(f"N-trend fit points used: {n_trend_fit.num_points_used}")
    _print_residual_summary_by_degree(
        scan_result,
        scaled_energy_by_n_trend,
        residual_analysis_by_degree,
    )

    raw_fig = plot_scan_raw_energy_surface(scan_result)
    raw_fig.axes[0].set_title("Raw total energy surface")

    n2_fig = plot_scan_normalized_energy_surface(scan_result, normalized_energy_n2)
    n2_fig.axes[0].set_title("Normalized energy surface: E / N^2")

    scaled_fig = plot_scan_normalized_energy_surface(scan_result, scaled_energy_by_n_trend)
    scaled_fig.axes[0].set_title("First-stage scaled surface: E / E_trend(N)")
    scaled_fig.axes[0].set_zlabel("E / E_trend(N)")

    for degree in _RESIDUAL_BASELINE_DEGREES:
        residual_surface = residual_analysis_by_degree[degree]["residual_surface"]
        residual_fig = plot_scan_residual_surface(
            scan_result,
            residual_surface,
            baseline_degree=degree,
        )
        residual_fig.axes[0].set_title(
            f"Final residual: E / E_trend(N) minus degree-{degree} baseline"
        )
        residual_fig.axes[0].set_zlabel(
            f"residual after degree-{degree} baseline subtraction"
        )
    plt.show()


def main_4() -> None:
    scan_specs = [
        # {
        #     "label": "family 13",
        #     "num_ions_vert": 1,
        #     "num_ions_hor": 3,
        #     "scale": 1.0,
        #     "alpha_min": 1.85,
        #     "alpha_max": 2.15,
        #     "num_alphas": 5,
        #     "num_ions_free_min": 4,
        #     "num_ions_free_max": 23,
        #     "best_of_k": 1,
        #     "seed_scale": 1.0,
        #     "vertical_seed_scale_factor": 2.0,
        #     "seed_jitter": 0.0,
        #     "rng_seed": 7,
        #     "basinhopping_niter": 40,
        # },
        # {
        #     "label": "family 14",
        #     "num_ions_vert": 1,
        #     "num_ions_hor": 4,
        #     "scale": 1.0,
        #     "alpha_min": 1.85,
        #     "alpha_max": 2.15,
        #     "num_alphas": 5,
        #     "num_ions_free_min": 4,
        #     "num_ions_free_max": 23,
        #     "best_of_k": 1,
        #     "seed_scale": 1.0,
        #     "vertical_seed_scale_factor": 2.0,
        #     "seed_jitter": 0.0,
        #     "rng_seed": 7,
        #     "basinhopping_niter": 40,
        # },
        {
            "label": "family 15",
            "num_ions_vert": 1,
            "num_ions_hor": 5,
            "scale": 1.0,
            "alpha_min": 1.85,
            "alpha_max": 2.15,
            "num_alphas": 3,
            "num_ions_free_min": 4,
            "num_ions_free_max": 23,
            "best_of_k": 1,
            "seed_scale": 1.0,
            "vertical_seed_scale_factor": 2.0,
            "seed_jitter": 0.0,
            "rng_seed": 7,
            "basinhopping_niter": 40,
        },
        {
            "label": "family 16",
            "num_ions_vert": 1,
            "num_ions_hor": 6,
            "scale": 1.0,
            "alpha_min": 1.85,
            "alpha_max": 2.15,
            "num_alphas": 3,
            "num_ions_free_min": 4,
            "num_ions_free_max": 23,
            "best_of_k": 1,
            "seed_scale": 1.0,
            "vertical_seed_scale_factor": 2.0,
            "seed_jitter": 0.0,
            "rng_seed": 7,
            "basinhopping_niter": 40,
        },
        {
            "label": "family 17",
            "num_ions_vert": 1,
            "num_ions_hor": 7,
            "scale": 1.0,
            "alpha_min": 1.85,
            "alpha_max": 2.15,
            "num_alphas": 3,
            "num_ions_free_min": 4,
            "num_ions_free_max": 23,
            "best_of_k": 1,
            "seed_scale": 1.0,
            "vertical_seed_scale_factor": 2.0,
            "seed_jitter": 0.0,
            "rng_seed": 7,
            "basinhopping_niter": 40,
        },
    ]

    completed_scans: list[dict[str, object]] = []

    for scan_index, scan_spec in enumerate(scan_specs, start=1):
        label = str(scan_spec["label"])
        print()
        print(f"=== running scan {scan_index}/{len(scan_specs)}: {label} ===")

        scan_result = scan_symmetry_family_over_alpha_and_free(
            num_ions_vert=int(scan_spec["num_ions_vert"]),
            num_ions_hor=int(scan_spec["num_ions_hor"]),
            scale=float(scan_spec["scale"]),
            alpha_min=float(scan_spec["alpha_min"]),
            alpha_max=float(scan_spec["alpha_max"]),
            num_alphas=int(scan_spec["num_alphas"]),
            num_ions_free_min=int(scan_spec["num_ions_free_min"]),
            num_ions_free_max=int(scan_spec["num_ions_free_max"]),
            best_of_k=int(scan_spec["best_of_k"]),
            seed_scale=float(scan_spec["seed_scale"]),
            vertical_seed_scale_factor=float(scan_spec["vertical_seed_scale_factor"]),
            seed_jitter=float(scan_spec["seed_jitter"]),
            rng_seed=None if scan_spec["rng_seed"] is None else int(scan_spec["rng_seed"]),
            basinhopping_niter=int(scan_spec["basinhopping_niter"]),
        )

        normalized_energy_n2 = normalize_scan_energy_by_n2(scan_result)
        median_energy_by_n = compute_median_energy_by_total_ion_count(scan_result)
        n_trend_fit = fit_energy_n_trend_power_law(median_energy_by_n)
        scaled_energy_by_n_trend = normalize_scan_energy_by_fitted_n_trend(scan_result, n_trend_fit)
        residual_analysis_by_degree = _compute_residual_analysis_by_degree(
            scan_result,
            scaled_energy_by_n_trend,
        )

        completed_scans.append(
            {
                "label": label,
                "scan_result": scan_result,
                "normalized_energy_n2": normalized_energy_n2,
                "scaled_energy_by_n_trend": scaled_energy_by_n_trend,
                "n_trend_fit": n_trend_fit,
                "residual_analysis_by_degree": residual_analysis_by_degree,
            }
        )

        successful_count = int(np.count_nonzero(scan_result.success_mask))
        total_count = int(scan_result.success_mask.size)
        mean_runtime = float(np.nanmean(scan_result.runtime_surface_s))
        print(
            "completed scan: "
            f"success={successful_count}/{total_count}, "
            f"mean runtime={mean_runtime:.3f} s, "
            f"N-trend p={n_trend_fit.p:.6g}"
        )
        _print_residual_summary_by_degree(
            scan_result,
            scaled_energy_by_n_trend,
            residual_analysis_by_degree,
        )

    print()
    print("All scans completed. Building plots.")

    for completed_scan in completed_scans:
        label = str(completed_scan["label"])
        scan_result = completed_scan["scan_result"]
        normalized_energy_n2 = completed_scan["normalized_energy_n2"]
        scaled_energy_by_n_trend = completed_scan["scaled_energy_by_n_trend"]
        residual_analysis_by_degree = completed_scan["residual_analysis_by_degree"]

        raw_fig = plot_scan_raw_energy_surface(scan_result)
        raw_fig.axes[0].set_title(f"{label}: raw total energy surface")

        normalized_fig = plot_scan_normalized_energy_surface(scan_result, normalized_energy_n2)
        normalized_fig.axes[0].set_title(f"{label}: normalized energy surface E / N^2")

        scaled_fig = plot_scan_normalized_energy_surface(scan_result, scaled_energy_by_n_trend)
        scaled_fig.axes[0].set_title(f"{label}: scaled energy surface E / E_trend(N)")
        scaled_fig.axes[0].set_zlabel("E / E_trend(N)")

        for degree in _RESIDUAL_BASELINE_DEGREES:
            residual_surface = residual_analysis_by_degree[degree]["residual_surface"]
            residual_fig = plot_scan_residual_surface(
                scan_result,
                residual_surface,
                baseline_degree=degree,
            )
            residual_fig.axes[0].set_title(
                f"{label}: final residual surface after degree-{degree} baseline"
            )
            residual_fig.axes[0].set_zlabel(
                f"residual after degree-{degree} baseline subtraction"
            )

    plt.show()


def main_5() -> None:
    comparison = compare_symmetry_seed_to_unrestricted_refinement(
        num_ions_vert=1,
        num_ions_hor=6,
        num_ions_free=19,
        alpha=1.95,
        scale=1.0,
        symmetry_solver_kwargs={
            "seed_scale": 1.0,
            "vertical_seed_scale_factor": 3.0,
            "seed_jitter": 0.0,
            "rng_seed": 7,
            "basinhopping_niter": 40,
        },
        unrestricted_solver_kwargs={
            "use_basinhopping": True,
            "local_maxiter": 4000,
        },
    )

    print("Symmetry seed to unrestricted harmonic refinement")
    print(f"total ion count: {int(comparison.total_ion_count)}")
    print(f"symmetry energy: {float(comparison.symmetry_energy):.10g}")
    print(f"unrestricted energy: {float(comparison.unrestricted_energy):.10g}")
    print(f"delta energy: {float(comparison.delta_energy):.10g}")
    print(f"sorted RMS displacement: {float(comparison.rms_displacement_sorted):.6g}")
    print(
        "unrestricted gradient max norm: "
        f"{float(comparison.unrestricted_gradient_max_norm):.3e}"
    )
    print(f"unrestricted success: {bool(comparison.unrestricted_success)}")
    print(f"unrestricted message: {comparison.unrestricted_message}")

    _plot_symmetry_vs_unrestricted_comparison(comparison)
    plt.show()


if __name__ == "__main__":
    main_5()
