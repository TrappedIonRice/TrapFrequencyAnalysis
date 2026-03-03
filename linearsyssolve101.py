# A module that builds a local linear control model for the trap by mapping electrode
# settings to a fitted polynomial potential, then using analytic derivatives of that
# polynomial to compute equilibrium constraints and mode properties.
#
# Core goal here: build a matrix A such that c = A @ u,
# where u = [V_DC1, V_DC2, ..., V_DCK, s] with s = V_rf^2 / omega_mhz^2,
# and omega_mhz = (2*pi*f_rf_hz)/1e6.
# and c are the polynomial coefficients
# (ordered as in PolynomialFeatures(include_bias=True)).

from __future__ import annotations

from typing import List, Tuple, Dict, Iterable
import time
import numpy as np

from sim.simulation import Simulation
import constants
from trapping_variables import Trapping_Vars
from trap_A_cache import DEFAULT_CACHE_DIR, get_or_build_A


def _normalize_bounds(bounds, n: int) -> List[Tuple[float, float]]:
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2 and not isinstance(bounds[0], (list, tuple)):
        return [tuple(bounds)] * n
    if isinstance(bounds, (list, tuple)) and len(bounds) == n:
        return [tuple(b) for b in bounds]
    raise ValueError("bounds must be (low, high) or a list of (low, high) per electrode")


def _generate_input_points(
    num_samples: int,
    dc_bounds: List[Tuple[float, float]],
    rf_dc_bounds: List[Tuple[float, float]],
    s_bounds: Tuple[float, float],
    seed: int = 0,
    include_axes: bool = True,
    include_zero: bool = True,
) -> np.ndarray:
    """
    Generate input samples u = [V_DC..., s], where s = V_rf^2 / omega_mhz^2
    and omega_mhz = (2*pi*f_rf_hz)/1e6.
    Includes optional zero and per-axis +/- points, plus random samples.
    """
    rng = np.random.default_rng(seed)
    k = len(dc_bounds)
    k_rf = len(rf_dc_bounds)

    samples = []
    if include_zero:
        samples.append(np.zeros(k + k_rf + 1, dtype=float))

    if include_axes:
        for i in range(k):
            lo, hi = dc_bounds[i]
            samples.append(_unit_with_value(k + k_rf + 1, i, lo))
            samples.append(_unit_with_value(k + k_rf + 1, i, hi))
        for i in range(k_rf):
            lo, hi = rf_dc_bounds[i]
            idx = k + i
            samples.append(_unit_with_value(k + k_rf + 1, idx, lo))
            samples.append(_unit_with_value(k + k_rf + 1, idx, hi))
        # s axis
        samples.append(_unit_with_value(k + k_rf + 1, k + k_rf, s_bounds[0]))
        samples.append(_unit_with_value(k + k_rf + 1, k + k_rf, s_bounds[1]))

    # randoms
    n_rand = max(0, num_samples - len(samples))
    if n_rand > 0:
        dc_lo = np.array([b[0] for b in dc_bounds], dtype=float)
        dc_hi = np.array([b[1] for b in dc_bounds], dtype=float)
        rf_dc_lo = np.array([b[0] for b in rf_dc_bounds], dtype=float)
        rf_dc_hi = np.array([b[1] for b in rf_dc_bounds], dtype=float)
        dc_rand = rng.uniform(dc_lo, dc_hi, size=(n_rand, k))
        rf_dc_rand = rng.uniform(rf_dc_lo, rf_dc_hi, size=(n_rand, k_rf))
        s_rand = rng.uniform(s_bounds[0], s_bounds[1], size=(n_rand, 1))
        rand = np.hstack([dc_rand, rf_dc_rand, s_rand])
        samples.extend(list(rand))

    return np.asarray(samples, dtype=float)


def _unit_with_value(n: int, idx: int, val: float) -> np.ndarray:
    v = np.zeros(n, dtype=float)
    v[idx] = float(val)
    return v


def _build_trapping_vars_from_u(
    dc_electrodes: List[str],
    rf_dc_electrodes: List[str],
    u: np.ndarray,
    rf_electrodes: Tuple[str, str] = ("RF1", "RF2"),
) -> Trapping_Vars:
    """
    Build Trapping_Vars for a given input vector u = [V_DC..., s].
    RF amplitude is sqrt(s) * omega_ref_mhz applied symmetrically to rf_electrodes.
    """
    k = len(dc_electrodes)
    k_rf = len(rf_dc_electrodes)
    if u.shape[0] != k + k_rf + 1:
        raise ValueError("u must have length K + K_rf_dc + 1")

    tv = Trapping_Vars()
    # DC electrodes
    for el, v in zip(dc_electrodes, u[:k]):
        tv.set_amp(tv.dc_key, el, float(v))
    # DC on RF electrodes
    for el, v in zip(rf_dc_electrodes, u[k : k + k_rf]):
        tv.set_amp(tv.dc_key, el, float(v))

    # RF drive from s = V_rf^2 / omega_mhz^2 (use reference frequency)
    s = float(u[-1])
    if s < 0:
        raise ValueError("s must be >= 0")
    omega_ref_mhz = float(constants.RF_OMEGA_REF_MHZ)  # rad/s scaled to MHz units
    rf_amp = float(np.sqrt(s) * omega_ref_mhz)
    f_ref = float(constants.RF_FREQ_REF_HZ)
    tv.add_driving("RF", f_ref, 0.0, {rf_electrodes[0]: rf_amp, rf_electrodes[1]: rf_amp})
    return tv


def build_voltage_to_c_matrix(
    trap_name: str,
    dc_electrodes: List[str],
    num_samples: int,
    rf_dc_electrodes: List[str] = ("RF1", "RF2"),
    dc_bounds: Iterable[Tuple[float, float]] = (-500.0, 500.0),
    rf_dc_bounds: Iterable[Tuple[float, float]] = (-50.0, 50.0),
    s_bounds: Tuple[float, float] = (0.0, constants.RF_S_MAX_DEFAULT),
    polyfit_deg: int = 4,
    seed: int = 0,
    rel_rmse_max: float = 1e-3,
    r2_min: float = 0.999,
) -> Dict[str, object]:
    """
    Build a matrix A such that c = A @ u where u = [V_DC..., s].

    Steps:
    - Generate input samples u across the specified bounds.
    - For each u, update trapping vars in a single Simulation instance.
    - Read c via sim.get_center_poly_c_vector.
    - Solve least squares for A.

    Returns dict with:
      - A: (M x (K+K_rf+1)) matrix mapping u -> c
      - powers: (M x 3) monomial powers for c ordering
      - samples_u: (N x (K+K_rf+1)) input samples
      - samples_c: (N x M) output coefficients
    """
    dc_bounds_list = _normalize_bounds(dc_bounds, len(dc_electrodes))
    rf_dc_bounds_list = _normalize_bounds(rf_dc_bounds, len(rf_dc_electrodes))
    u_samples = _generate_input_points(
        num_samples=num_samples,
        dc_bounds=dc_bounds_list,
        rf_dc_bounds=rf_dc_bounds_list,
        s_bounds=s_bounds,
        seed=seed,
    )

    # Single simulation instance
    sim = Simulation(trap_name, Trapping_Vars())

    c_samples = []
    powers_ref = None
    for i, u in enumerate(u_samples):
        t0 = time.time()
        print(f"point {i}")
        tv = _build_trapping_vars_from_u(
            dc_electrodes=dc_electrodes,
            rf_dc_electrodes=rf_dc_electrodes,
            u=u,
        )
        sim.change_electrode_variables(tv)
        sim.clear_held_results()
        c, powers = sim.get_center_poly_c_vector(polyfit_deg=polyfit_deg, return_powers=True)
        if powers_ref is None:
            powers_ref = powers
        c_samples.append(c)
        print(f"point {i} time: {time.time() - t0:.3f}s")

    C = np.asarray(c_samples, dtype=float)  # (N x M)
    U = np.asarray(u_samples, dtype=float)  # (N x K+1)

    # Solve U @ X = C for X ((K+K_rf+1) x M) => A = X.T (M x (K+K_rf+1))
    X, residuals, rank, svals = np.linalg.lstsq(U, C, rcond=None)
    A = X.T

    # Linearity check and error stuff
    C_pred = U @ X
    err = C_pred - C
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sqrt(np.mean(C**2))) if np.any(C) else float("nan")
    rel_rmse = float(rmse / denom) if denom and not np.isnan(denom) else float("nan")

    # Per-coefficient r^2
    c_mean = np.mean(C, axis=0)
    ss_res = np.sum((C - C_pred) ** 2, axis=0)
    ss_tot = np.sum((C - c_mean) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_per_c = 1.0 - (ss_res / ss_tot)

    # Nonlinearity checks
    nonlin_flags = []
    if rel_rmse_max is not None and rel_rmse > rel_rmse_max:
        nonlin_flags.append(f"fit_rel_rmse {rel_rmse:.3e} > {rel_rmse_max:.3e}")
    if r2_min is not None:
        bad_r2 = np.where(r2_per_c < r2_min)[0]
        if bad_r2.size > 0:
            nonlin_flags.append(
                f"r2_per_c below {r2_min:.3f} for {bad_r2.size} coefficients"
            )
    if nonlin_flags:
        print("[warn] Nonlinearity indicators: " + "; ".join(nonlin_flags))

    return {
        "A": A,
        "powers": powers_ref,
        "samples_u": U,
        "samples_c": C,
        "rf_dc_electrodes": list(rf_dc_electrodes),
        "s_bounds": s_bounds,
        "u_layout": {
            "dc": (0, len(dc_electrodes)),
            "rf_dc": (len(dc_electrodes), len(dc_electrodes) + len(rf_dc_electrodes)),
            "s": (len(dc_electrodes) + len(rf_dc_electrodes), len(dc_electrodes) + len(rf_dc_electrodes) + 1),
        },
        # "fit_rmse": rmse,
        "fit_rel_rmse": rel_rmse,
        # "fit_residuals": residuals,
        # "fit_rank": rank,
        # "fit_svals": svals,
        "fit_r2_per_c": r2_per_c,
        "nonlinearity_flags": nonlin_flags,
    }


def build_voltage_to_c_matrix_cached(
    trap_name: str,
    dc_electrodes: List[str],
    num_samples: int,
    rf_dc_electrodes: List[str] = ("RF1", "RF2"),
    dc_bounds: Iterable[Tuple[float, float]] = (-500.0, 500.0),
    rf_dc_bounds: Iterable[Tuple[float, float]] = (-50.0, 50.0),
    s_bounds: Tuple[float, float] = (0.0, constants.RF_S_MAX_DEFAULT),
    polyfit_deg: int = 4,
    seed: int = 0,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_rebuild: bool = False,
) -> Dict[str, object]:
    return get_or_build_A(
        cache_dir=cache_dir,
        trap_name=trap_name,
        dc_electrodes=dc_electrodes,
        rf_dc_electrodes=rf_dc_electrodes,
        polyfit_deg=polyfit_deg,
        dc_bounds=dc_bounds,
        rf_dc_bounds=rf_dc_bounds,
        s_bounds=s_bounds,
        num_samples=num_samples,
        seed=seed,
        builder_fn=build_voltage_to_c_matrix,
        force_rebuild=force_rebuild,
    )


if __name__ == "__main__":
    out = build_voltage_to_c_matrix(
        trap_name="1252dTrapRice",
        dc_electrodes=[f"DC{i}" for i in range(1, 21)],
        num_samples=50,
    )
    print("A shape:", out["A"].shape)
    print("A:", out["A"])
    print("fit_rel_rmse:", out["fit_rel_rmse"])

    out2 = build_voltage_to_c_matrix(
        trap_name="InnTrapFine",
        dc_electrodes=[f"DC{i}" for i in range(1, 13)],
        num_samples=50,
    )
    print("A shape:", out2["A"].shape)
    print("A:", out2["A"])
    print("fit_rel_rmse:", out2["fit_rel_rmse"])

    out3 = build_voltage_to_c_matrix(
        trap_name="Simp58_101",
        dc_electrodes=[f"DC{i}" for i in range(1, 11)],
        num_samples=50,
    )
    print("A shape:", out3["A"].shape)
    print("A:", out3["A"])
    print("fit_rel_rmse:", out3["fit_rel_rmse"])
