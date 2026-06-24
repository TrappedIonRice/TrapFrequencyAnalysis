"""
basic_forward_sim_example.py

Minimal forward simulation: given trap voltages, compute the single-ion
equilibrium position, secular frequencies, and principal axis directions.

Run from the repo root:
    python simulation_highlevel_implementation_files/basic_forward_sim_example.py
"""

import sys
import os

# Allow imports from the repo root regardless of where this script is called from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.simulation import Simulation
from trapping_variables import Trapping_Vars


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Trap selection
    # -------------------------------------------------------------------------
    # "Simp58_101"              — 10 DC electrodes (DC1–DC10) + RF1, RF2
    # "2Dtrap_125_45deg_200exp" — 20 DC electrodes (DC1–DC20) + RF1, RF2
    #
    # The string must match a subfolder under Data/ that contains
    # combined_dataframe.csv (pre-built from COMSOL exports).
    TRAP = "Simp58_101"

    # -------------------------------------------------------------------------
    # Trapping variables — Simp58_101
    # -------------------------------------------------------------------------
    tv = Trapping_Vars()

    # RF drive: tv.add_driving(label, frequency_Hz, phase_rad, {electrode: amplitude_V})
    # Both RF blades are driven at the same amplitude for a symmetric pseudopotential.
    tv.add_driving("RF", 43_000_000, 0.0, {"RF1": 892.76, "RF2": 892.76})

    # DC voltages via structured helpers:
    #   add_twist_dc(twist)  — adds `twist` V to all DC* electrodes, subtracts from RF1/RF2
    #                          on the DC drive; rotates the transverse potential
    #   add_endcaps_dc(v)    — adds `v` V to the axial endcap electrodes (DC1, DC5, DC6, DC10)
    #                          to provide axial confinement
    tv.add_twist_dc(3.275)
    tv.add_endcaps_dc(5.0)

    # DC bias on the RF blades (RF1, RF2 on the DC drive).
    # Non-zero values shift the ion equilibrium in the transverse (y-z) plane.
    tv.set_amp(tv.dc_key, "RF1", -3.0)
    tv.set_amp(tv.dc_key, "RF2", -3.0)

    # # -------------------------------------------------------------------------
    # # [ALTERNATE] Set every DC electrode voltage explicitly instead of using helpers.
    # # Uncomment this block and comment out the two helper calls above.
    # # -------------------------------------------------------------------------
    # dc_map = {
    #     "DC1": -0.31, "DC2": 4.77, "DC3": -1.22, "DC4": 4.77, "DC5": -0.26,
    #     "DC6": -0.31, "DC7": 4.77, "DC8": -1.22, "DC9": 4.77, "DC10": -0.26,
    #     "RF1": -7.74, "RF2": -7.74,  # DC bias on RF blades
    # }
    # for electrode, volts in dc_map.items():
    #     tv.set_amp(tv.dc_key, electrode, volts)

    # -------------------------------------------------------------------------
    # [ALTERNATE TRAP] 2Dtrap_125_45deg_200exp — 20 DC electrodes + RF1, RF2
    # Uncomment this block and set TRAP = "2Dtrap_125_45deg_200exp" above.
    # -------------------------------------------------------------------------
    # tv = Trapping_Vars()
    # tv.add_driving("RF", 43_000_000, 0.0, {"RF1": 655.94, "RF2": 655.94})
    #
    # add_endcaps_mid_center_5E applies three voltage levels across the 20 DC electrodes:
    #   endcaps — DC1,5,6,10,11,15,16,20  (axial ends)
    #   mid     — DC3,8,13,18             (axial midpoints)
    #   center  — DC2,4,7,9,12,14,17,19  (transverse center ring)
    # tv.add_endcaps_mid_center_5E(endcaps=100.8, mid=-15.0, center=54.0)

    # Or set every electrode (including RF DC bias) explicitly:
    # dc_map_2d = {
    #     "DC1":  64.61, "DC2":  35.33, "DC3":  -15.0, "DC4":  34.07, "DC5":  64.45,
    #     "DC6":  64.26, "DC7":  33.92, "DC8":  -15.0, "DC9":  35.18, "DC10": 64.74,
    #     "DC11": 64.37, "DC12": 34.11, "DC13": -15.0, "DC14": 35.16, "DC15": 64.67,
    #     "DC16": 64.66, "DC17": 35.22, "DC18": -15.0, "DC19": 33.85, "DC20": 64.33,
    #     "RF1":  17.66, "RF2":  17.66,  # DC bias on RF blades
    # }
    # for electrode, volts in dc_map_2d.items():
    #     tv2.set_amp(tv2.dc_key, electrode, volts)

    # -------------------------------------------------------------------------
    # Build the simulation and run the full forward pipeline
    # -------------------------------------------------------------------------
    sim = Simulation(TRAP, tv)

    # _smoke_test_new_stack does in sequence:
    #   1. Compute total voltage and pseudopotential columns from electrode amplitudes
    #   2. Fit a degree-4 polynomial to the potential in the trap center region
    #   3. Minimize the fitted potential to find the ion equilibrium position
    #   4. Build and diagonalize the Hessian → secular frequencies and mode vectors
    #   5. Contract 3rd/4th-order derivatives into mode coupling tensors (g3, g4)
    sim._smoke_test_new_stack(n_ions=1, poly_deg=4)

    # Project the lab-frame mode vectors onto the principal axes of the potential.
    # For a single ion the principal directions are simply the eigenvectors of
    # the 3×3 Hessian, which is why this step requires n_ions == 1 to have run first.
    sim.compute_principal_directions_from_one_ion()
    sim.populate_normalmodes_in_prinipledir_freq_labels()

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------
    eq_pos = sim.ion_equilibrium_positions.get(1)       # shape (1, 3), metres
    ppack  = sim.principal_dir_normalmodes_andfrequencies.get(1)
    freqs  = ppack.get("frequencies_Hz")                # 3 secular frequencies in Hz
    pdirs  = sim.principal_dirs                         # 3×3 array — rows are principal directions
    # expressed in lab (x, y, z) coordinates

    print("\n--- Single-ion results ---")
    print("Equilibrium position (m):", eq_pos)
    print("Secular frequencies (Hz):", freqs)
    print("Principal axis directions (row i = direction i in lab x,y,z):")
    print(pdirs)
