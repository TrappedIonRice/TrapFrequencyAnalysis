"""
Dummy inverse-design runner for InnTrapFine.

This is a minimal, hard-coded example that wires the optimization together.
"""

from __future__ import annotations

import numpy as np

from inverse_design import solve_u_for_targets
from sim.simulation import Simulation
from trapping_variables import Trapping_Vars
import time
import constants


def run_demo() -> None:
    start_time = time.time()
    # Target specification (hard-coded demo values)
    r0 = np.array([0.0, 0.0, 0.0], dtype=float)
    freqs = np.array(
        [0.360e6, 1.910e6, 2.320e6], dtype=float
    )  # Hz
    principal_axis = [1.0, 0.0, 0.0]
    ref_dir = [0.0, 1.0, 0.0]
    alpha_deg =30

    # Trap/fit configuration for InnTrapFine
    trap_name = "Simp58_101"
    dc_electrodes = [f"DC{i}" for i in range(1, 11)]

    u_bounds = (
        [(-1000, 1000)] * 10  # DC1..DC10
        + [(-1000.00000, 1000.00000)] * 2  # RF1_DC, RF2_DC
        + [(0.0, constants.RF_S_MAX_DEFAULT)]  # s = V^2 / omega^2
    )

    # Demo forward-check uses reference frequency to match A-model.
    rf_freq_hz = constants.RF_FREQ_REF_HZ

    out = solve_u_for_targets(
        r0=r0,
        freqs=freqs,
        principal_axis=principal_axis,
        ref_dir=ref_dir,
        alpha_deg=alpha_deg,
        ion_mass_kg=constants.ion_mass,  # example: 40Ca+ in kg
        ion_charge_c=1.602176634e-19,
        poly_is_potential_energy=False,
        freqs_in_hz=True,
        trap_name=trap_name,
        dc_electrodes=dc_electrodes,
        num_samples=20,
        s_bounds=(0.0, constants.RF_S_MAX_DEFAULT),
        polyfit_deg=4,
        objective="l2_dc",
        s_penalty_scale=1e-7,
        enforce_bounds=False,
        u_bounds=u_bounds,
    )

    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken (seconds):", total_time)

    np.set_printoptions(precision=6, suppress=True)
    print("status:", out["status"])
    print("u:", out["u"])
    print("residual_norm:", out["residual_norm"])
    print("u_norm2:", out["u_norm2"])
    print("u_norminf:", out["u_norminf"])
    print("s:", out["u_s"])
    if out["u"] is None or out["status"] != "ok":
        print("Solve failed; solver_info:", out["solver_info"])
        return

    # Run simulation with the found u to inspect principal directions and freqs
    u = out["u"]
    s = float(out["u_s"])
    # s = V_rf^2 / omega_mhz^2 with omega_mhz = (2*pi*f_rf_hz)/1e6, so rf_amp is in volts.
    rf_amp = float(np.sqrt(max(s, 0.0)) * constants.RF_OMEGA_REF_MHZ)
    print("rfamp: ", rf_amp)
    print("rf_freq_hz: ", rf_freq_hz)
    tv = Trapping_Vars()
    for el, v in zip(dc_electrodes, u[: len(dc_electrodes)]):
        tv.set_amp(tv.dc_key, el, float(v))
    for el, v in zip(["RF1", "RF2"], u[len(dc_electrodes) : len(dc_electrodes) + 2]):
        tv.set_amp(tv.dc_key, el, float(v))
    tv.add_driving("RF", rf_freq_hz, 0.0, {"RF1": rf_amp, "RF2": rf_amp})
    sim = Simulation(trap_name, Trapping_Vars())
    sim.change_electrode_variables(tv)
    sim.clear_held_results()

    sim.find_equilib_position_single(num_ions=1)
    eq_pos = sim.ion_equilibrium_positions.get(1)
    print("Single-ion equilibrium position (m):", eq_pos)

    sim.get_static_normal_modes_and_freq(1, normalize=True, sort_by_freq=True)
    sim.compute_principal_directions_from_one_ion()
    sim.populate_normalmodes_in_prinipledir_freq_labels()
    ppack = sim.principal_dir_normalmodes_andfrequencies.get(1)
    if ppack is None:
        raise RuntimeError("Principal-direction data not populated for n=1.")
    freqs_hz = np.asarray(ppack.get("frequencies_Hz"), dtype=float)
    principal_dirs_found = np.asarray(sim.principal_dirs, dtype=float)

    print("Sim freqs (Hz):", freqs_hz)
    print("Sim principal dirs (rows are dirs):")
    print(principal_dirs_found)


if __name__ == "__main__":
    run_demo()
