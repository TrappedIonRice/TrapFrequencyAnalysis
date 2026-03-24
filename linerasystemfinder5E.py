# The idea of this file is to find the coefficoents of the linear system which takes
# the five inputs (RFAmp, RFDC, DCEndcaps, DCMIdcaps, DCCenter) onto the 
# 3 outputs (freq_x, freq_y, freq_z)

# To do this we will initialize a sim for given (trap, drive frequency, numpoints) and then test enough points to determine
# The coeffs of the linear system (IE fill in the matrix). 
# Speificaly numpoints will be tested, and then the coeffs will be found by solving the linear system (IE 2 is min)

# Further there will be a internal test within this procidure to check the assumption of linearity is valid.

# Note: For now we will assume the priciple directions allign with the lab axes, this is valid for the 2d trap and these inputs.
# However this assumption will make this procedure hard to generalize to other geometries or less symetric inputs.

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from sim.simulation import Simulation
from trapping_variables import Trapping_Vars


def _generate_input_points(num_points: int) -> List[Tuple[float, float, float, float, float]]:
    """
    Generate a list of 5-tuples (RFAmp, RFDC, DCEndcaps, DCMidcaps, DCCenter).

    Convention: start at 0.1 for all inputs, then increase by a decade for each
    subsequent point: 0.1, 1.0, 10.0, ...
    """
    if num_points < 1:
        raise ValueError("num_points must be >= 1")
    vals = [0.1 * (10 ** i) for i in range(num_points)]
    return [(v, v, v, v, v) for v in vals]


def _check_principal_dirs(principal_dirs: np.ndarray, tol: float = 0.9) -> bool:
    """
    Return True if principal directions align with lab axes within tolerance.
    Prints a warning if not aligned.
    """
    if principal_dirs.shape != (3, 3):
        print(f"[warn] principal_dirs has shape {principal_dirs.shape}, expected (3,3)")
        return False

    max_axes = []
    ok = True
    for i, v in enumerate(principal_dirs):
        v = np.asarray(v, dtype=float)
        idx = int(np.argmax(np.abs(v)))
        max_axes.append(idx)
        if np.abs(v[idx]) < tol:
            print(
                "[warn] principal dir not aligned to lab axis: "
                f"dir_{i}={v.tolist()}"
            )
            ok = False

    if len(set(max_axes)) != 3:
        print(
            "[warn] principal dirs do not map uniquely to x/y/z axes: "
            f"axis_indices={max_axes}"
        )
        ok = False
    return ok


def get_data_from_system_5E(
    trap_name: str,
    rf_freq_hz: float,
    num_points: int,
) -> List[Dict[str, object]]:
    """
    For each input point, build Trapping_Vars using 5E end/mid/center,
    apply RFamp to RF drive and RFDC to DC offsets on RF1/RF2, then run the
    simulation and return principal trapping frequencies.

    Returns a list of dicts with inputs, frequencies (Hz), and principal dirs.
    """
    input_points = _generate_input_points(int(num_points))

    # Initialize a single simulation instance (we will swap trapping variables)
    sim = Simulation(trap_name, Trapping_Vars())

    results: List[Dict[str, object]] = []
    for point in input_points:
        rf_amp, rf_dc, dc_end, dc_mid, dc_center = [float(x) for x in point]

        tv = Trapping_Vars()
        tv.add_driving(
            "RF",
            float(rf_freq_hz),
            0.0,
            {"RF1": rf_amp, "RF2": rf_amp},
        )
        tv.add_endcaps_mid_center_5E(dc_end, dc_mid, dc_center)

        # RFDC: add DC offset to both RF1/RF2 on DC drive
        ea_dc = tv.Var_dict[tv.dc_key]
        for el in ("RF1", "RF2"):
            if el in ea_dc.amplitudes:
                ea_dc.add_amplitude_volt(el, rf_dc)

        # Run simulation for this trapping vars
        sim.change_electrode_variables(tv)
        sim.clear_held_results()
        sim.find_equilib_position_single(num_ions=1, minimizertype="InitGuess")
        sim.get_static_normal_modes_and_freq(num_ions=1, normalize=True, sort_by_freq=True)
        sim.compute_principal_directions_from_one_ion()
        sim.populate_normalmodes_in_prinipledir_freq_labels()

        ppack = sim.principal_dir_normalmodes_andfrequencies.get(1)
        if ppack is None:
            raise RuntimeError("Principal-direction data not populated for n=1.")

        freqs_hz = np.asarray(ppack.get("frequencies_Hz"), dtype=float)
        principal_dirs = np.asarray(sim.principal_dirs, dtype=float)
        _check_principal_dirs(principal_dirs)

        results.append(
            {
                "inputs": {
                    "RFAmp": rf_amp,
                    "RFDC": rf_dc,
                    "DCEndcaps": dc_end,
                    "DCMidcaps": dc_mid,
                    "DCCenter": dc_center,
                },
                "principal_freqs_Hz": freqs_hz.tolist(),
                "principal_dirs": principal_dirs.tolist(),
            }
        )

    return results


    
    




