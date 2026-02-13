"""
Utilities to compute kappa from RF settings and secular frequencies.
"""



import math
from typing import Callable, Iterable
import numpy as np

from sim.simulation import Simulation
from trapping_variables import Trapping_Vars
import constants

def get_trapping_vars_from_ampandFreq(freq, amp):
        tv = Trapping_Vars()
        # RF drive setup
        rf_freq_hz = freq
        rf_amp_rf1 = amp
        rf_amp_rf2 = amp
        tv.add_driving("RF", rf_freq_hz, 0.0, {"RF1": rf_amp_rf1, "RF2": rf_amp_rf2})
        return tv


def get_secular_freqs_from_principal_dirs(sim: Simulation, num_ions: int = 1):
    """
    Return [f_y, f_z] where f_y is the max secular frequency aligned with +/‑y,
    and f_z is the max secular frequency aligned with +/‑z.
    """
    sim.get_static_normal_modes_and_freq(num_ions)
    sim.compute_principal_directions_from_one_ion()
    sim.populate_normalmodes_in_prinipledir_freq_labels()

    ppack = sim.principal_dir_normalmodes_andfrequencies.get(num_ions)
    if ppack is None:
        return [float("nan"), float("nan")]

    freqs = np.asarray(ppack.get("frequencies_Hz"), dtype=float)
    align = np.asarray(ppack.get("dir_alignment"), dtype=int)

    principal_dirs = np.asarray(sim.principal_dirs, dtype=float)
    if principal_dirs.shape != (3, 3):
        return [float("nan"), float("nan")]

    y_dir = int(np.argmax(np.abs(principal_dirs[:, 1])))
    z_dir = int(np.argmax(np.abs(principal_dirs[:, 2])))

    f_y = np.max(freqs[align == y_dir]) if np.any(align == y_dir) else float("nan")
    f_z = np.max(freqs[align == z_dir]) if np.any(align == z_dir) else float("nan")
    return [float(f_y), float(f_z)]


def get_secular_freq_fromAmpandFreq(sim: Simulation, freq, amp):
    trap_vars = get_trapping_vars_from_ampandFreq(freq, amp)
    sim.change_electrode_variables(trap_vars)
    sim.clear_held_results()
    sim.find_equilib_position_single(num_ions=1, minimizertype="InitGuess")
    return get_secular_freqs_from_principal_dirs(sim, num_ions=1)

def get_kappas_from_secularFreqs_amp_rffreq(freq, amp, secular_freqs):
    """Compute kappa from secular frequencies, RF amplitude and RF frequency.

    Args:
        freq (float): RF drive frequency in Hz.
        amp (float): RF drive amplitude in V.
        secular_freqs (list[float]): [f_y, f_z] secular frequencies in Hz.

    Returns:
        list[float]: [kappa_y, kappa_z].
    """
    print(secular_freqs)
    out = []
    for secular_freq in secular_freqs:
        if secular_freq is None or np.isnan(secular_freq):
            out.append(float("nan"))
            continue
        kappa = (
            ((2 * math.pi) * secular_freq)
            * math.sqrt(2)
            * (constants.ion_mass / constants.ion_charge)
            * ((2 * math.pi) * freq / (amp))
            * (0.000125**2)
            #* (0.00025**2)
        )
        out.append(kappa)
    return out

def calculate_kappa(sim: Simulation, rf_freq, rf_amp):
    secular_freqs = get_secular_freq_fromAmpandFreq(sim, rf_freq, rf_amp)
    kappas = get_kappas_from_secularFreqs_amp_rffreq(
        rf_freq, rf_amp, secular_freqs
    )
    return kappas


def compute_kappa_stats_over_grid(
    dataset: str,
    rf_freq_min: float,
    rf_freq_max: float,
    rf_freq_points: int,
    rf_amp_min: float,
    rf_amp_max: float,
    rf_amp_points: int,
):
    rf_freqs = np.linspace(float(rf_freq_min), float(rf_freq_max), int(rf_freq_points))
    rf_amps = np.linspace(float(rf_amp_min), float(rf_amp_max), int(rf_amp_points))

    sim = Simulation(dataset)
    kappas_y = []
    kappas_z = []

    for rf_freq in rf_freqs:
        for rf_amp in rf_amps:
            k_y, k_z = calculate_kappa(sim, rf_freq, rf_amp)
            kappas_y.append(k_y)
            kappas_z.append(k_z)

    kappas_y = np.asarray(kappas_y, dtype=float)
    kappas_z = np.asarray(kappas_z, dtype=float)

    stats = {
        "kappa_y_mean": float(np.nanmean(kappas_y)),
        "kappa_y_std": float(np.nanstd(kappas_y)),
        "kappa_z_mean": float(np.nanmean(kappas_z)),
        "kappa_z_std": float(np.nanstd(kappas_z)),
    }
    return stats

print(compute_kappa_stats_over_grid(
    "Comsol_125",
    rf_freq_min=10e6,
    rf_freq_max=60e6,
    rf_freq_points=3,
    rf_amp_min=100,
    rf_amp_max=700,
    rf_amp_points=3,
))


# sim = Simulation("Hyper_2")

# print(calculate_kappa(sim, 42e6, 560))

# print(calculate_kappa(sim, 50e6, 580))
# print(calculate_kappa(sim, 54e6, 540))
# print(calculate_kappa(sim, 56e6, 520))
# print(calculate_kappa(sim, 58e6, 500))
# print(calculate_kappa(sim, 60e6, 480))
# print(calculate_kappa(sim, 52e6, 460))
# print(calculate_kappa(sim, 48e6, 440))
