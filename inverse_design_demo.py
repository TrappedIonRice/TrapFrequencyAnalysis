"""
demo for inverse-design runner.

This is a minimal, hard-coded example that wires the inverse problem pipeline together.
"""

from __future__ import annotations

import numpy as np

from inverse_design import solve_u_for_exact_targets, solve_u_for_frequency_box
from sim.simulation import Simulation
from trapping_variables import Trapping_Vars
import time
import constants

# NOTE: The matrix A is built by sampling random inputs and fitting the linear system
# this takes quite a bit as each point means a run in the forward direction
# to cut down on time there is a system to cache the matrix A -- it is not full proof at the moment
# but id say its 95 percent full proof...
# anyway this A is saved and has a tag on it that identifies it, if you run this pipeline with the same tag
# then the stored A is used, otherwise a new A is built (takes a few minuites)
# this tag consists of the trap name, the number of dc_electrodes, the box in which the fits are done (from constants.py)
# a nondimensionalized unit, degree of polyfit
# if any of these change then a new A will be triggered, beware!
# to be clear making new A is not a bad thing, the file is small, the time is just a bit annoying


def run_demo() -> None:
    start_time = time.time()

    ### Target specification (hard-coded demo values) ###

    # Eq position you want
    r0 = np.array([0.0, 0.0, 0.0], dtype=float)

    freq_x = 450 #kHz
    freq_y = 2000 #kHz
    freq_z = 900 #kHz
    # Mode freq you want in Hz
    freqs = np.array([freq_x*1e3, freq_z*1e3, freq_y*1e3], dtype=float)  # Hz

    # Principle directions. "Princ_axis" defines the first principal direction,
    # and "ref_dir" "alpha_deg" are used to define the other two by:
    # The second principle direction is ref_dir projected onto the plane orthogonal to princ_axis,
    # and then rotated by alpha_deq away from this projection (right handed rotation)
    # The thrid is defined by the first two.
    principal_axis = [1.0, 0.0, 0.0]
    ref_dir = [0.0, 0.0, 1.0]
    alpha_deg = 0

    # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # *
    # the following defines which trap to use, its electrodes, and the bounds on the control voltages
    # Note that u is a row vector of the form [DC1, DC2, ..., RF1_DC, RF2_DC, s]
    # where s = V_rf^2 / omega_rf^2, this is the linear pseudopotential "strength" parameter
    # Below are three parameters

    # Trap/fit configuration for InnTrapFine
    # trap_name = "InnTrapFine"
    # dc_electrodes = [f"DC{i}" for i in range(1, 13)]

    # u_bounds = (
    #     [(-1000, 1000)] * 12  # DC1..DC10
    #     + [(-1000.00000, 1000.00000)] * 2  # RF1_DC, RF2_DC
    #     + [(0.0, constants.RF_S_MAX_DEFAULT)]  # s = V^2 / omega^2
    # )

    # # Trap/fit configuration for 1252dTrapRIce
    # trap_name = "1252dTrapRice"
    # dc_electrodes = [f"DC{i}" for i in range(1, 21)]

    # u_bounds = (
    #     [(-1000, 1000)] * 20  # DC1..DC10
    #     + [(-1000.00000, 1000.00000)] * 2  # RF1_DC, RF2_DC
    #     + [(0.0, constants.RF_S_MAX_DEFAULT)]  # s = V^2 / omega^2
    # )

    # # Trap/fit configuration for Simp58_101
    # trap_name = "Simp58_101"
    # dc_electrodes = [f"DC{i}" for i in range(1, 11)]

    # u_bounds = (
    #     [(-1000, 1000)] * 10  # DC1..DC10
    #     + [(-1000.00000, 1000.00000)] * 2  # RF1_DC, RF2_DC
    #     + [(0.0, constants.RF_S_MAX_DEFAULT)]  # s = V^2 / omega^2
    # )

    # # Trap/fit configuration for 2Dtrap_125_45deg_200exp
    trap_name = "2D trap V4.4.125 - c - 75deg 150um ground_MORE_exposed_0.1_DC_With_RF_284_+_curv"
    dc_electrodes = [f"DC{i}" for i in range(1, 21)]

    u_bounds = (
        [(-30, 70)] * 20  # DC1..DC10
        + [(-20.00000, 20.00000)] * 2  # RF1_DC, RF2_DC
        + [(0.0, constants.RF_S_MAX_DEFAULT*100)]  # s = V^2 / omega^2
    )

    # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # *

    # Demo forward-check uses reference frequency to match A-model.

    # This is just the ref_hz, what you make it
    # only changes the presentation of the result when s split back into RFamp and RFfreq
    rf_freq_hz = constants.RF_FREQ_REF_HZ

    # This is the minimization used:
    min_objective = "l2_dc"  # minimize L2 norm of DC voltages this ignores s
    # your options are: "l2", "linf", "weighted_l2", "avg_max_dc", "l2_dc"
    # l2 minimizes L2 norm of all
    # linf minimizes L-inf norm of all
    # weighted_l2 minimizes a weighted L2 norm of all, with s weighted by s_penalty_scale
    # avg_max_dc minimizes the average of the max static v on the DC and the max static v on the RF
    # IMPORTANT: Just use l2_dc and allow bounds to check s

    # wheather to take into acount the given bounds or not
    enforce_bounds_on_u = False

    # the call of the inverse pipeline
    out = solve_u_for_exact_targets(
        r0=r0,
        freqs=freqs,
        principal_axis=principal_axis,
        ref_dir=ref_dir,
        alpha_deg=alpha_deg,
        ion_mass_kg=constants.ion_mass,
        ion_charge_c=constants.ion_charge,
        poly_is_potential_energy=False,
        freqs_in_hz=True,
        trap_name=trap_name,
        dc_electrodes=dc_electrodes,
        num_samples=80,  # just how many sample to sample when defining A
        s_bounds=(0.0, constants.RF_S_MAX_DEFAULT),
        polyfit_deg=6,
        objective=min_objective,
        s_penalty_scale=1e-7,  # not used
        enforce_bounds=enforce_bounds_on_u,
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

    # Run simulation with the found u to compare principal directions and freqs
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


def run_demo_box() -> None:
    """
    Demo for frequency-box mode:
      - mode 1 exact
      - modes 2/3 only lower bounded (positive floor)
    """
    start_time = time.time()

    r0 = np.array([0.0, 0.0, 0.0], dtype=float)

    # Define 1 example use
    # freq_floor_hz = 0.100e6
    # freq_bounds = [
    #     (0.300e6, 0.300e6),
    #     (freq_floor_hz, None),
    #     (freq_floor_hz, None),
    # ]

    # Define 2 example use
    freq_floor_hz = 0.100e6
    freq_bounds = [
        (0.300e6, 0.300e6),
        (0.600e6, 00.600e6),
        (freq_floor_hz, None),
    ]

    # Define all example use
    # freq_floor_hz = 0.100e6
    # freq_bounds = [
    #     (0.250e6, 0.40e6),
    #     (0.4e6, 1.2e6),
    #     (1.8e6, 5e6),
    # ]

    # Define 1box example use
    # freq_floor_hz = 0.100e6
    # freq_bounds = [
    #     (0.300e6, 0.300e6),
    #     (0.300e6, 1.500e6),
    #     (2e6, 3e6),
    # ]

    principal_axis = [1.0, 0.0, 0.0]
    ref_dir = [0.0, 0.0, 1.0]
    alpha_deg = 0

    trap_name = "InnTrapFine"
    dc_electrodes = [f"DC{i}" for i in range(1, 13)]
    u_bounds = (
        [(-100, 100)] * 12
        + [(-100.0000, 100.00000)] * 2
        + [(0.0, constants.RF_S_MAX_DEFAULT)]
    )

    out = solve_u_for_frequency_box(
        r0=r0,
        freq_bounds=freq_bounds,
        principal_axis=principal_axis,
        ref_dir=ref_dir,
        alpha_deg=alpha_deg,
        ion_mass_kg=constants.ion_mass,
        ion_charge_c=constants.ion_charge,
        poly_is_potential_energy=False,
        freqs_in_hz=True,
        trap_name=trap_name,
        dc_electrodes=dc_electrodes,
        num_samples=80,
        s_bounds=(0.0, constants.RF_S_MAX_DEFAULT),
        polyfit_deg=4,
        objective="l2_dc",
        s_penalty_scale=1e-7,
        enforce_bounds=False,
        u_bounds=u_bounds,
    )

    end_time = time.time()
    print("Total time taken (seconds):", end_time - start_time)
    np.set_printoptions(precision=6, suppress=True)
    print("status:", out["status"])
    print("u:", out["u"])
    print("residual_norm:", out["residual_norm"])

    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken (seconds):", total_time)

    np.set_printoptions(precision=6, suppress=True)
    print("status:", out["status"])
    print("u:", out["u"])
    print("ineq_violation:", out["solver_info"].get("ineq_violation"))
    print("ineq_tol:", out["solver_info"].get("ineq_tol"))
    print("residual_norm:", out["residual_norm"])
    print("u_norm2:", out["u_norm2"])
    print("u_norminf:", out["u_norminf"])
    print("s:", out["u_s"])
    if out["u"] is None or out["status"] != "ok":
        print("Solve failed; solver_info:", out["solver_info"])
        return

    rf_freq_hz = constants.RF_FREQ_REF_HZ
    # Run simulation with the found u to compare principal directions and freqs
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


def demo_1():

    start_time = time.time()

    r0 = np.array([0.0, 0.0, 0.0], dtype=float)

    # freqs = np.array([0.300e6, 0.600e6, 1.900e6], dtype=float)  # Hz
    # Create a grid of frequency values
    freq_min = 0.1e6  # Hz
    freq_max = 4.0e6  # Hz
    num_freqs = 23
    frequencies_list = []
    frequencies_list = [
        np.array(combo, dtype=float)
        for combo in np.ndindex((num_freqs,) * 3)
        for combo in [np.linspace(freq_min, freq_max, num_freqs)[np.array(combo)]]
    ]
    print("Number of frequency combinations:", len(frequencies_list))

    # Principle directions. "Princ_axis" defines the first principal direction,
    # and "ref_dir" "alpha_deg" are used to define the other two by:
    # The second principle direction is ref_dir projected onto the plane orthogonal to princ_axis,
    # and then rotated by alpha_deq away from this projection (right handed rotation)
    # The thrid is defined by the first two.
    principal_axis = [1.0, 0.0, 0.0]
    ref_dir = [0.0, 0.0, 1.0]
    alpha_deg = 0

    # Trap/fit configuration for InnTrapFine
    trap_name = "InnTrapFine"
    dc_electrodes = [f"DC{i}" for i in range(1, 13)]

    u_bounds = (
        [(-100, 100)] * 12  # DC1..DC10
        + [(-100.00000, 100.00000)] * 2  # RF1_DC, RF2_DC
        + [(0.0, constants.RF_S_MAX_DEFAULT)]  # s = V^2 / omega^2
    )

    # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # * # *

    rf_freq_hz = constants.RF_FREQ_REF_HZ

    # This is the minimization used:
    min_objective = "l2_dc"  # minimize L2 norm of DC voltages this ignores s

    enforce_bounds_on_u = True

    outs = []

    for freqs in frequencies_list:
        # the call of the inverse pipeline
        out = solve_u_for_exact_targets(
            r0=r0,
            freqs=freqs,
            principal_axis=principal_axis,
            ref_dir=ref_dir,
            alpha_deg=alpha_deg,
            ion_mass_kg=constants.ion_mass,
            ion_charge_c=constants.ion_charge,
            poly_is_potential_energy=False,
            freqs_in_hz=True,
            trap_name=trap_name,
            dc_electrodes=dc_electrodes,
            num_samples=80,  # just how many sample to sample when defining A
            s_bounds=(0.0, constants.RF_S_MAX_DEFAULT),
            polyfit_deg=4,
            objective=min_objective,
            s_penalty_scale=1e-7,  # not used
            enforce_bounds=enforce_bounds_on_u,
            u_bounds=u_bounds,
        )
        outs.append(out)

    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken (seconds):", total_time)

    pass


if __name__ == "__main__":
    run_demo()
    # run_demo_box()
    # demo_1()
