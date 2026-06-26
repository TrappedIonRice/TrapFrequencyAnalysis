"""
voltage_sensitivity_analysis.py

Monte Carlo analysis of how DC voltage uncertainty propagates into micromotion
(MM) amplitude and other single-ion operating-point quantities.

Algorithm
---------
For each of N_SAMPLES draws:
  1. Add independent Gaussian noise (σ = DELTA_V volts) to every DC electrode
     voltage on the existing Simulation object — no re-initialization.
  2. Rerun the single-ion forward pipeline (poly fit → equilibrium → modes).
  3. Evaluate MM amplitude at the perturbed equilibrium position.
After all samples, compute statistics and plot a 20-bucket histogram.

Run from the repo root:
    python simulation_highlevel_implementation_files/voltage_sensitivity_analysis.py
"""

import io
import sys
import os
import contextlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.simulation import Simulation
from trapping_variables import Trapping_Vars
import constants


# =============================================================================
# Configuration
# =============================================================================
TRAP      = "Simp58_101"
DELTA_V   = 0.005    # Gaussian σ applied independently to each DC electrode (volts)
N_SAMPLES = 1000     # number of Monte Carlo draws
POLY_DEG  = 4
RNG_SEED  = np.random.randint(0, 2**31)


def _build_nominal_tv() -> Trapping_Vars:
    """Nominal operating point"""
    tv = Trapping_Vars()
    tv.add_driving("RF", 25_500_000, 0.0, {"RF1": 600, "RF2": 600})
    tv.add_twist_dc(3.275)
    tv.add_endcaps_dc(3)
    tv.set_amp(tv.dc_key, "RF1", 0)
    tv.set_amp(tv.dc_key, "RF2", 0)
    return tv


# =============================================================================
# MM amplitude at a single 3-D point
# =============================================================================

def get_MM_amplitude_at_point(sim: "Simulation", xyz_m, e_interp=None) -> float:
    """
    Return micromotion amplitude (metres) at position xyz_m.

    Formula: MM = (q / m Ω²) |E_RF(r)|

    Parameters
    ----------
    sim      : Simulation with total_voltage_df already loaded.
    xyz_m    : array-like (3,), equilibrium position in metres.
    e_interp : optional LinearNDInterpolator (xyz -> |E_RF|) pre-built over the
               center region.  When provided, E is computed by smooth trilinear
               interpolation rather than nearest-grid-point lookup, eliminating
               the step-function discretisation artifact.  Falls back to
               nearest-grid-point if the point is outside the interpolation hull
               (returns NaN) or if e_interp is None.
    """
    rf_drives = [dk for dk in sim.trapVariables.get_drives() if dk.f_uHz != 0]
    if not rf_drives:
        raise ValueError("No RF drive found — cannot compute micromotion.")
    drive = max(rf_drives, key=lambda d: d.f_hz)
    omega = 2.0 * np.pi * drive.f_hz
    const = constants.ion_charge / (constants.ion_mass * omega ** 2)

    if e_interp is not None:
        e_mag = float(e_interp([xyz_m]))   # NaN if outside convex hull
        if not np.isnan(e_mag):
            return const * e_mag
        # outside the interpolation region — fall through to nearest-grid-point

    ex, ey, ez = sim.get_total_E_components(drive=drive)
    df = sim.total_voltage_df
    x0, y0, z0 = float(xyz_m[0]), float(xyz_m[1]), float(xyz_m[2])
    dist_sq = (
        (df["x"].to_numpy() - x0) ** 2
        + (df["y"].to_numpy() - y0) ** 2
        + (df["z"].to_numpy() - z0) ** 2
    )
    idx   = int(np.argmin(dist_sq))
    e_mag = np.sqrt(float(ex[idx]) ** 2 + float(ey[idx]) ** 2 + float(ez[idx]) ** 2)
    return const * e_mag


# =============================================================================
# Monte Carlo sweep
# =============================================================================

def run_voltage_sensitivity_analysis(
    sim: "Simulation",
    delta_v: float = DELTA_V,
    n_samples: int = N_SAMPLES,
    seed: int = RNG_SEED,
):
    """
    Vary each DC electrode voltage independently (Gaussian σ = delta_v),
    run a single-ion forward simulation for each sample, and return
    MM amplitude statistics, an operating-point uncertainty box, and a plot.

    The Simulation object is mutated per sample (voltages changed in place via
    set_amp) then restored to nominal on return.  The expensive dataframe load
    happens only once (at Simulation construction).

    Parameters
    ----------
    sim       : Simulation — already initialized with the nominal TV.
    delta_v   : float — Gaussian σ per electrode in volts.
    n_samples : int   — number of Monte Carlo draws.
    seed      : int   — RNG seed for reproducibility.

    Returns
    -------
    stats : dict
        Keys: min, max, mean, median, q1, q3, std, n_valid  (all in metres / Hz).
    box : dict
        Keys: x_m, y_m, z_m, f0_Hz, f1_Hz, f2_Hz.
        Each value is a (lo, hi) tuple giving the observed range.
    fig : matplotlib.figure.Figure
    """
    rng    = np.random.default_rng(seed)
    dc_key = sim.trapVariables.dc_key

    # Snapshot nominal DC amplitudes (all electrodes in the DC drive, incl. RF DC biases)
    nominal_v = {
        el: sim.trapVariables.get_amp(dc_key, el)
        for el in sim.trapVariables.Var_dict[dc_key].amplitudes
    }
    dc_electrodes = list(nominal_v.keys())

    # ------------------------------------------------------------------ #
    # One-time pre-computation: eliminate both the sklearn refit and the  #
    # numexpr update_total_voltage_columns from the per-sample loop.      #
    #                                                                      #
    # The combined potential is linear in electrode voltages:             #
    #   V_total(r) = Σ_el A_el * V_el(r)  +  V_pseudo(r)                 #
    # so its polynomial coefficient vector is:                            #
    #   c_total = Σ_el A_el * c_el  +  c_pseudo                           #
    # We pre-fit c_el for every electrode and c_pseudo for the RF term.   #
    # Per sample we replace the costly sklearn fit with a dot product.    #
    # ------------------------------------------------------------------ #
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sim.update_total_voltage_columns()
        sim.update_center_polys(polyfit_deg=POLY_DEG)

    # DC model object is stored by reference — mutating coef_ updates it in-place
    # everywhere evaluate_center_poly reads it without any dict surgery.
    dc_model, dc_poly_transformer, _ = sim.center_fits[dc_key]

    # Center-region point coordinates (used only for the pre-computation)
    df = sim.total_voltage_df
    cx = constants.center_region_x_um * 1e-6
    cy = constants.center_region_y_um * 1e-6
    cz = constants.center_region_z_um * 1e-6
    center_mask = (
        df["x"].between(-cx, cx)
        & df["y"].between(-cy, cy)
        & df["z"].between(-cz, cz)
    )
    cutout_xyz = df.loc[center_mask, ["x", "y", "z"]].values  # (N_center, 3)
    X_poly     = dc_poly_transformer.transform(cutout_xyz)     # (N_center, 35)

    # Per-electrode polynomial coefficients
    coef_per_electrode = {}
    for el in dc_electrodes:
        col = f"{el}_V"
        if col in df.columns:
            v_el = df.loc[center_mask, col].values
            coef_per_electrode[el] = np.linalg.lstsq(X_poly, v_el, rcond=None)[0]
        else:
            coef_per_electrode[el] = np.zeros(X_poly.shape[1])

    # RF pseudopotential coefficient vector:
    #   pseudo(r) = Static_TotalV(r) - Σ_el A_el_nominal * V_el(r)
    dc_scalar_nom = sum(
        nominal_v[el] * df.loc[center_mask, f"{el}_V"].values
        for el in dc_electrodes if f"{el}_V" in df.columns
    )
    pseudo_center = df.loc[center_mask, "Static_TotalV"].values - dc_scalar_nom
    coef_pseudo   = np.linalg.lstsq(X_poly, pseudo_center, rcond=None)[0]

    # Stack electrode coefficient vectors into a matrix for a fast matmul per sample
    # shape: (N_electrodes, N_poly_terms)
    el_list   = list(coef_per_electrode.keys())
    coef_mat  = np.vstack([coef_per_electrode[el] for el in el_list])  # (Ne, Np)

    # Pre-build the RF E-field interpolator once.
    # RF amplitude is fixed for the entire sweep so ex_rf/ey_rf/ez_rf never change.
    # LinearNDInterpolator builds a Delaunay triangulation over the center-region
    # mesh points and evaluates by barycentric interpolation — continuous and exact
    # on the mesh, no grid-snapping discontinuities.
    rf_drives = [dk for dk in sim.trapVariables.get_drives() if dk.f_uHz != 0]
    rf_drive  = max(rf_drives, key=lambda d: d.f_hz)
    ex_rf, ey_rf, ez_rf = sim.get_total_E_components(drive=rf_drive)
    mask_np   = center_mask.to_numpy()
    e_mag_center = np.sqrt(ex_rf[mask_np]**2 + ey_rf[mask_np]**2 + ez_rf[mask_np]**2)
    print("Building RF E-field interpolator over center region…", end=" ", flush=True)
    e_interp  = LinearNDInterpolator(cutout_xyz, e_mag_center)
    print("done.")

    mm_values  = []
    eq_pos_arr = []
    freq_arr   = []
    n_failed   = 0

    print(f"Running {n_samples} Monte Carlo samples  (δV = {delta_v:.4f} V σ per electrode)…")

    for i in range(n_samples):
        # Sample perturbed amplitudes
        amp_vec = np.array([nominal_v[el] + rng.normal(0.0, delta_v) for el in el_list])

        # New polynomial coefficients via dot product — replaces the entire
        # update_total_voltage_columns + get_voltage_poly_for_drive_at_region path.
        dc_model.coef_[:] = coef_pseudo + amp_vec @ coef_mat

        # Keep trapVariables in sync (the optimizer reads electrode amps indirectly
        # through the polynomial, but we update for bookkeeping consistency).
        for el, a in zip(el_list, amp_vec):
            sim.trapVariables.set_amp(dc_key, el, float(a))

        # Clear only the computed quantities that depend on the potential.
        # center_fits is intentionally NOT cleared — dc_model.coef_ was updated above.
        sim.ion_equilibrium_positions.clear()
        sim.ion_eigenvectors.clear()
        sim.ion_eigenvalues.clear()
        sim.normal_modes_and_frequencies.clear()
        sim.driven_g_0_2_couplings.clear()
        sim.driven_g_0_3_couplings.clear()
        sim.inherent_g_0_3_couplings.clear()
        sim.inherent_g_0_4_couplings.clear()
        sim.principal_dir_normalmodes_andfrequencies.clear()
        sim.principal_dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                sim.find_equilib_position_single(num_ions=1)
                sim.get_static_normal_modes_and_freq(1)
        except Exception as exc:
            print(f"  sample {i:4d}: pipeline failed — {exc}")
            n_failed += 1
            continue

        eq = sim.ion_equilibrium_positions.get(1)
        if eq is None:
            n_failed += 1
            continue

        xyz = eq[0]   # shape (3,), metres
        mm  = get_MM_amplitude_at_point(sim, xyz, e_interp=e_interp)

        nmf   = sim.normal_modes_and_frequencies.get(1)
        freqs = np.asarray(nmf["frequencies_Hz"]) if nmf is not None else np.full(3, np.nan)

        mm_values.append(mm)
        eq_pos_arr.append(xyz.copy())
        freq_arr.append(freqs)
        print(i)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1:4d}/{n_samples}  ({n_failed} failed so far)")

    # Restore nominal voltages
    for el, v in nominal_v.items():
        sim.trapVariables.set_amp(dc_key, el, v)

    mm_arr  = np.array(mm_values)
    pos_arr = np.array(eq_pos_arr)   # (n_valid, 3)
    frq_arr = np.array(freq_arr)     # (n_valid, 3)

    print(f"Done. {len(mm_arr)} valid samples, {n_failed} failed.")

    # --- MM statistics ---
    stats = {
        "min":     float(np.nanmin(mm_arr)),
        "max":     float(np.nanmax(mm_arr)),
        "mean":    float(np.nanmean(mm_arr)),
        "median":  float(np.nanmedian(mm_arr)),
        "q1":      float(np.nanpercentile(mm_arr, 25)),
        "q3":      float(np.nanpercentile(mm_arr, 75)),
        "std":     float(np.nanstd(mm_arr)),
        "n_valid": len(mm_arr),
    }

    # --- Uncertainty box: observed range of each operating-point quantity ---
    box = {}
    if len(pos_arr):
        box["x_m"]   = (float(np.nanmin(pos_arr[:, 0])), float(np.nanmax(pos_arr[:, 0])))
        box["y_m"]   = (float(np.nanmin(pos_arr[:, 1])), float(np.nanmax(pos_arr[:, 1])))
        box["z_m"]   = (float(np.nanmin(pos_arr[:, 2])), float(np.nanmax(pos_arr[:, 2])))
    if len(frq_arr):
        box["f0_Hz"] = (float(np.nanmin(frq_arr[:, 0])), float(np.nanmax(frq_arr[:, 0])))
        box["f1_Hz"] = (float(np.nanmin(frq_arr[:, 1])), float(np.nanmax(frq_arr[:, 1])))
        box["f2_Hz"] = (float(np.nanmin(frq_arr[:, 2])), float(np.nanmax(frq_arr[:, 2])))

    fig = _plot_mm_histogram(mm_arr, stats, delta_v)
    return stats, box, fig, pos_arr


# =============================================================================
# Histogram with statistics overlay
# =============================================================================

def _plot_mm_histogram(mm_arr, stats, delta_v):
    """
    20-bucket bar chart of normalized MM amplitude distribution.
    Bucket width = (max - min) / 20.  y-axis = fraction of valid samples.
    """
    lo, hi    = stats["min"], stats["max"]
    bin_edges = np.linspace(lo, hi, 21)           # 21 edges → 20 buckets
    counts, _ = np.histogram(mm_arr, bins=bin_edges)
    norm_counts = counts / stats["n_valid"]
    centers     = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width       = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    ax.bar(
        centers * 1e9, norm_counts, width=width * 1e9,
        color="steelblue", edgecolor="black", linewidth=0.5,
    )

    ax.axvline(stats["median"] * 1e9, color="red",    lw=2,
               label=f'median = {stats["median"]*1e9:.3f} nm')
    ax.axvline(stats["mean"]   * 1e9, color="orange", lw=2, linestyle="--",
               label=f'mean   = {stats["mean"]*1e9:.3f} nm')
    ax.axvspan(
        stats["q1"] * 1e9, stats["q3"] * 1e9,
        alpha=0.2, color="green", label="IQR (Q1–Q3)",
    )

    text_lines = [
        f'N        = {stats["n_valid"]}',
        f'δV       = {delta_v:.4f} V  (σ)',
        f'min      = {stats["min"]*1e9:.3f} nm',
        f'max      = {stats["max"]*1e9:.3f} nm',
        f'mean     = {stats["mean"]*1e9:.3f} nm',
        f'std      = {stats["std"]*1e9:.3f} nm',
        f'Q1       = {stats["q1"]*1e9:.3f} nm',
        f'Q3       = {stats["q3"]*1e9:.3f} nm',
    ]
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.97, 0.97, "\n".join(text_lines),
        transform=ax.transAxes, fontsize=9,
        va="top", ha="right", bbox=props, family="monospace",
    )

    ax.set_xlabel("MM amplitude (nm)")
    ax.set_ylabel("Fraction of samples per bucket")
    ax.set_title(
        f"MM amplitude distribution  —  N={stats['n_valid']},  "
        f"σ_V = {delta_v:.4f} V / electrode"
    )
    ax.legend(loc="upper left")
    return fig


def _plot_position_histograms(pos_arr):
    """Three 20-bucket histograms of equilibrium x, y, z positions across all samples."""
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    for ax, col, label in zip(axes, range(3), labels):
        vals_um = pos_arr[:, col] * 1e6
        lo, hi  = vals_um.min(), vals_um.max()
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        bins    = np.linspace(lo, hi, 21)
        counts, _ = np.histogram(vals_um, bins=bins)
        norm    = counts / len(pos_arr)
        centers = 0.5 * (bins[:-1] + bins[1:])
        width   = bins[1] - bins[0]

        ax.bar(centers, norm, width=width, color="steelblue", edgecolor="black", linewidth=0.5)
        ax.axvline(np.mean(vals_um),   color="orange", lw=2, linestyle="--",
                   label=f"mean = {np.mean(vals_um):+.4f} µm")
        ax.axvline(np.median(vals_um), color="red",    lw=2,
                   label=f"med  = {np.median(vals_um):+.4f} µm")
        ax.set_xlabel(f"{label} (µm)")
        ax.set_ylabel("Fraction of samples")
        ax.set_title(f"Equilibrium {label}  (σ = {np.std(vals_um)*1e3:.2f} nm)")
        ax.legend(fontsize=8)

    fig.suptitle(f"Equilibrium position distributions  (N={len(pos_arr)}, δV={DELTA_V:.4f} V σ)")
    return fig


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    nominal_tv = _build_nominal_tv()
    sim = Simulation(TRAP, nominal_tv)

    stats, box, fig, pos_arr = run_voltage_sensitivity_analysis(
        sim, delta_v=DELTA_V, n_samples=N_SAMPLES
    )

    # --- Print equilibrium positions for every sample ---
    print(f"\n{'='*60}")
    print(f"=== Equilibrium Positions  ({len(pos_arr)} samples) ===")
    print(f"{'='*60}")
    print(f"{'Sample':>7}   {'x (um)':>12}   {'y (um)':>12}   {'z (um)':>12}")
    for i, pos in enumerate(pos_arr):
        print(f"  {i:4d}   {pos[0]*1e6:>+12.4f}   {pos[1]*1e6:>+12.4f}   {pos[2]*1e6:>+12.4f}")

    # --- Print MM statistics ---
    print(f"\n{'='*60}")
    print("=== MM Amplitude Statistics ===")
    print(f"{'='*60}")
    for k, v in stats.items():
        if k == "n_valid":
            print(f"  {k:<12}: {v}")
        else:
            print(f"  {k:<12}: {v*1e9:.4f} nm")

    # --- Print uncertainty box ---
    print(f"\n{'='*60}")
    print("=== Operating-Point Uncertainty Box ===")
    print(f"{'='*60}")
    for k, (lo, hi) in box.items():
        if k.endswith("_m"):
            lo_um, hi_um = lo * 1e6, hi * 1e6
            print(f"  {k:<8}: [{lo_um:+.4f},  {hi_um:+.4f}]  um")
        else:
            lo_mhz, hi_mhz = lo * 1e-6, hi * 1e-6
            print(f"  {k:<8}: [{lo_mhz:.4f},  {hi_mhz:.4f}]  MHz")

    fig_pos = _plot_position_histograms(pos_arr)
    plt.show()
