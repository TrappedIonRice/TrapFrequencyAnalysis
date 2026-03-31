from __future__ import annotations

"""Ion-count scan entry point for the reduced-model equilibrium playground."""

from dataclasses import dataclass
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from equilibrium_playground.cryo2d_closecopy_2 import solve_cryo2d_closecopy_2
from equilibrium_playground.viewer import build_result_figure, print_result_summary
from equilibrium_playground.wrappers import best_of_k_cryo2d_closecopy_2


@dataclass(frozen=True)
class ScanPoint:
    num_ions: int
    energy: float
    solve_time_s: float
    success: bool
    message: str
    gradient_max_norm: float


def _main2_coefficients() -> dict[str, float]:
    """Return the fixed coefficients currently used in `main2`."""

    return {
        "a20": 0.89,
        "a02": 3.5,
        "a40": 0.0,
        "a22": 0.0,
        "a04": 0.0,
        "jitter": 0.1,
    }


def run_ion_count_scan(
    *,
    n_min: int = 5,
    n_max: int = 25,
) -> list[ScanPoint]:
    """Run `solve_cryo2d_closecopy_2` once for each ion count in `n_min..n_max`."""

    coeffs = _main2_coefficients()
    results: list[ScanPoint] = []
    for num_ions in range(int(n_min), int(n_max) + 1):
        t_start = perf_counter()
        result = solve_cryo2d_closecopy_2(
            num_ions,
            coeffs["a20"],
            coeffs["a02"],
            coeffs["a40"],
            coeffs["a22"],
            coeffs["a04"],
            jitter=coeffs["jitter"],
        )
        elapsed_s = perf_counter() - t_start
        results.append(
            ScanPoint(
                num_ions=num_ions,
                energy=float(result.energy),
                solve_time_s=float(elapsed_s),
                success=bool(result.success),
                message=str(result.message),
                gradient_max_norm=float(result.projected_gradient_max_norm),
            )
        )
    return results


def _quadratic_fit(x: np.ndarray, y: np.ndarray) -> np.poly1d:
    """Fit a quadratic trend to the measured data."""

    return np.poly1d(np.polyfit(x, y, deg=2))


def print_scan_results(scan_results: list[ScanPoint]) -> None:
    """Print the measured energy and solve time for each ion count."""

    coeffs = _main2_coefficients()
    print("Cryo2d_closecopy_2 ion-count scan")
    print(
        "Coefficients: "
        f"a20={coeffs['a20']}, "
        f"a02={coeffs['a02']}, "
        f"a40={coeffs['a40']}, "
        f"a22={coeffs['a22']}, "
        f"a04={coeffs['a04']}, "
        f"jitter={coeffs['jitter']}"
    )
    print()
    print(
        f"{'N':>3}  {'energy':>14}  {'time_s':>10}  {'success':>7}  "
        f"{'max|grad|':>12}  message"
    )
    for point in scan_results:
        print(
            f"{point.num_ions:3d}  "
            f"{point.energy:14.8f}  "
            f"{point.solve_time_s:10.4f}  "
            f"{str(point.success):>7}  "
            f"{point.gradient_max_norm:12.3e}  "
            f"{point.message}"
        )


def plot_scan_results(scan_results: list[ScanPoint]):
    """Plot measured energies and solve times with quadratic trend fits."""

    if not scan_results:
        return None

    num_ions = np.asarray([point.num_ions for point in scan_results], dtype=float)
    energies = np.asarray([point.energy for point in scan_results], dtype=float)
    solve_times = np.asarray([point.solve_time_s for point in scan_results], dtype=float)

    energy_fit = _quadratic_fit(num_ions, energies)
    time_fit = _quadratic_fit(num_ions, solve_times)
    num_ions_dense = np.linspace(num_ions.min(), num_ions.max(), 400)

    print()
    print("Energy quadratic fit:")
    print(
        "E(N) ~= "
        f"{energy_fit.c[0]:.6g} N^2 + {energy_fit.c[1]:.6g} N + {energy_fit.c[2]:.6g}"
    )
    print("Solve-time quadratic fit:")
    print(
        "t(N) ~= "
        f"{time_fit.c[0]:.6g} N^2 + {time_fit.c[1]:.6g} N + {time_fit.c[2]:.6g}"
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.5), constrained_layout=True)

    axes[0].plot(num_ions, energies, "o", color="#1f77b4", label="measured energy")
    axes[0].plot(
        num_ions_dense,
        energy_fit(num_ions_dense),
        "-",
        color="#d62728",
        label="quadratic trend",
    )
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Cryo2d_closecopy_2 energy vs ion count")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(num_ions, solve_times, "o", color="#2ca02c", label="measured solve time")
    axes[1].plot(
        num_ions_dense,
        time_fit(num_ions_dense),
        "-",
        color="#ff7f0e",
        label="quadratic trend",
    )
    axes[1].set_xlabel("Number of ions")
    axes[1].set_ylabel("Solve time (s)")
    axes[1].set_title("Cryo2d_closecopy_2 solve time vs ion count")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    return fig


def main() -> None:
    scan_results = run_ion_count_scan()
    print_scan_results(scan_results)
    fig = plot_scan_results(scan_results)
    if fig is not None:
        plt.show()


def main2() -> None:
    """Run the reduced-model close-copy solver 10 times at N=47."""

    coeffs = _main2_coefficients()
    num_ions = 47
    repeated_result = best_of_k_cryo2d_closecopy_2(
        num_ions,
        coeffs["a20"],
        coeffs["a02"],
        coeffs["a40"],
        coeffs["a22"],
        coeffs["a04"],
        k=10,
        jitter=coeffs["jitter"],
    )

    print("Best of 10 runs")
    print_result_summary(repeated_result.best_result, wrapper_result=repeated_result)
    print()

    fig = build_result_figure(
        repeated_result.best_result,
        title="Reduced even model: best of 10 Cryo-style runs at N=47",
    )
    if fig is not None:
        plt.show()


if __name__ == "__main__":
    main()
