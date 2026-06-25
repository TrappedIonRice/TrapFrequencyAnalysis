"""
multi_ion_sim_example.py

Forward simulation for an n-ion chain: equilibrium positions, 3n secular
frequencies, the full normal mode matrix, and per-mode per-ion participation.

Run from the repo root:
    python simulation_highlevel_implementation_files/multi_ion_sim_example.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.simulation import Simulation
from trapping_variables import Trapping_Vars

# Number of ions — change this to run a different chain length.
N_IONS = 13


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Trap selection  (same options as basic_forward_sim_example.py)
    # -------------------------------------------------------------------------
    TRAP = "Simp58_101"

    # -------------------------------------------------------------------------
    # Trapping variables  (identical setup to basic_forward_sim_example.py)
    # -------------------------------------------------------------------------
    tv = Trapping_Vars()

    tv.add_driving("RF", 25_500_000, 0.0, {"RF1": 536.0, "RF2": 536.0})

    # DC voltages via structured helpers:
    #   add_twist_dc(twist)  — adds `twist` V to all DC electrodes, subtracts from RF1/RF2
    #                          on the DC drive; rotates the transverse potential
    #   add_endcaps_dc(v)    — adds `v` V to the axial endcap electrodes (DC1, DC5, DC6, DC10)
    #                          to provide axial confinement
    tv.add_twist_dc(-2)
    tv.add_endcaps_dc(3)

    # DC bias on the RF blades (RF1, RF2 on the DC drive).
    # tv.set_amp(tv.dc_key, "RF1", 2)
    # tv.set_amp(tv.dc_key, "RF2", 2)



    # # DC voltages — set each electrode explicitly.
    # # To use the structured helpers instead, comment this block out and
    # # uncomment the add_twist_dc / add_endcaps_dc calls in basic_forward_sim_example.py.
    # dc_map = {
    #     "DC1": -0.31, "DC2":  4.77, "DC3": -1.22, "DC4":  4.77, "DC5": -0.26,
    #     "DC6": -0.31, "DC7":  4.77, "DC8": -1.22, "DC9":  4.77, "DC10": -0.26,
    #     "RF1": -7.74, "RF2": -7.74,   # DC bias on RF blades
    # }

    # for electrode, volts in dc_map.items():
    #     tv.set_amp(tv.dc_key, electrode, volts)

    # -------------------------------------------------------------------------
    # [ALTERNATE TRAP] 2Dtrap_125_45deg_200exp — 20 DC electrodes + RF1, RF2
    # Set TRAP = "2Dtrap_125_45deg_200exp" above and uncomment one block below.
    # -------------------------------------------------------------------------
    # tv = Trapping_Vars()
    # tv.add_driving("RF", 43_000_000, 0.0, {"RF1": 655.94, "RF2": 655.94})
    # tv.add_endcaps_mid_center_5E(endcaps=100.8, mid=-15.0, center=54.0)

    # Or every electrode explicitly (values from main_2 in simulation.py):
    # dc_map_2d = {
    #     "DC1":  64.61, "DC2":  35.33, "DC3":  -15.0, "DC4":  34.07, "DC5":  64.45,
    #     "DC6":  64.26, "DC7":  33.92, "DC8":  -15.0, "DC9":  35.18, "DC10": 64.74,
    #     "DC11": 64.37, "DC12": 34.11, "DC13": -15.0, "DC14": 35.16, "DC15": 64.67,
    #     "DC16": 64.66, "DC17": 35.22, "DC18": -15.0, "DC19": 33.85, "DC20": 64.33,
    #     "RF1":  17.66, "RF2":  17.66,
    # }
    # for electrode, volts in dc_map_2d.items():
    #     tv.set_amp(tv.dc_key, electrode, volts)

    # -------------------------------------------------------------------------
    # Build the simulation and run the forward pipeline
    # -------------------------------------------------------------------------
    sim = Simulation(TRAP, tv)

    # Step 1 — single-ion run: establishes the polynomial fit and finds the
    # principal axis directions of the trap potential.  The N-ion run reuses
    # the cached polynomial, and its mode labels use these principal dirs.
    sim._smoke_test_new_stack(n_ions=1, poly_deg=4)
    sim.compute_principal_directions_from_one_ion()
    sim.populate_normalmodes_in_prinipledir_freq_labels()

    # Step 2 — N-ion run: finds all N-ion equilibrium positions, diagonalizes
    # the 3N×3N Hessian, and projects modes onto the principal-dir frame above.
    sim._smoke_test_new_stack(n_ions=N_IONS, poly_deg=4)

    # -------------------------------------------------------------------------
    # Extract results
    # -------------------------------------------------------------------------
    eq_pos    = sim.ion_equilibrium_positions.get(N_IONS)   # (N_IONS, 3), metres
    ppack     = sim.principal_dir_normalmodes_andfrequencies.get(N_IONS)
    freqs     = np.asarray(ppack["frequencies_Hz"])         # (3*N_IONS,), Hz
    modes     = np.asarray(ppack["modes"])                  # (3*N_IONS, 3*N_IONS)
    dir_align = np.asarray(ppack["dir_alignment"])          # (3*N_IONS,), values 0/1/2

    # dir_alignment values 0/1/2 refer to the single-ion principal axes (dir_0,
    # dir_1, dir_2), whose lab-frame orientations are in sim.principal_dirs.
    AXIS = {0: "dir_0", 1: "dir_1", 2: "dir_2"}

    # =========================================================================
    # === 0. Single-ion results ===============================================
    # =========================================================================
    eq_pos_1 = sim.ion_equilibrium_positions.get(1)
    ppack_1  = sim.principal_dir_normalmodes_andfrequencies.get(1)
    freqs_1  = np.asarray(ppack_1["frequencies_Hz"])
    pdirs    = sim.principal_dirs   # shape (3, 3); row i = principal dir i in lab x,y,z

    print(f"\n{'='*60}")
    print("=== Single-Ion Results ===")
    print(f"{'='*60}")
    print("Equilibrium position (m):", eq_pos_1)
    print("Secular frequencies (Hz):", freqs_1)
    print("Principal axis directions (row i = direction i in lab x,y,z):")
    print(pdirs)

    # =========================================================================
    # === 1. Equilibrium Positions ============================================
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"=== Equilibrium Positions  ({N_IONS} ions) ===")
    print(f"{'='*60}")
    print(f"{'Ion':>4}   {'x (m)':>16}   {'y (m)':>16}   {'z (m)':>16}")
    for i, pos in enumerate(eq_pos):
        print(f"  {i:2d}   {pos[0]:>+16.6e}   {pos[1]:>+16.6e}   {pos[2]:>+16.6e}")

    # =========================================================================
    # === 2. Secular Frequencies + Axis Alignment =============================
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"=== Secular Frequencies  ({3 * N_IONS} modes) ===")
    print(f"{'='*60}")
    print(f"{'Mode':>4}   {'Frequency (Hz)':>18}   Aligns with")
    for i, (f, a) in enumerate(zip(freqs, dir_align)):
        print(f"  {i:2d}   {f:>18.2f}   {AXIS[int(a)]}-axis  (dir {int(a)})")

    # =========================================================================
    # === 3. Normal Mode Matrix ================================================
    # =========================================================================
    # Shape: (3N, 3N).  Columns are modes, sorted by frequency (col 0 = lowest).
    # Row layout: [x_ion0, y_ion0, z_ion0,  x_ion1, y_ion1, z_ion1,  ...]
    #   — for ion j: rows 3j, 3j+1, 3j+2 correspond to x, y, z
    # Entry [r, c]: participation of ion (r // 3) in direction (r % 3) for mode c.
    np.set_printoptions(precision=4, suppress=True, linewidth=220, threshold=10**6)
    print(f"\n{'='*60}")
    print(f"=== Normal Mode Matrix  {modes.shape} ===")
    print(f"{'='*60}")
    print("Columns = modes (0 = lowest freq).  Rows = ion coordinates.")
    print("Row r  ->  ion = r // 3,  direction = r % 3  (0=x, 1=y, 2=z)")
    print(modes)

    # =========================================================================
    # === 4. Per-Mode Participation (per ion) =================================
    # =========================================================================
    # For each mode (column of the mode matrix), reshape the 3N-vector into
    # an (N_IONS, 3) array.  Row i shows how ion i participates in [x, y, z].
    print(f"\n{'='*60}")
    print(f"=== Per-Mode Participation (per ion) ===")
    print(f"{'='*60}")
    print("Each block: rows = ions,  columns = [x, y, z] participation.")
    for mode_idx in range(3 * N_IONS):
        mode_vec = modes[:, mode_idx]
        per_ion  = mode_vec.reshape(N_IONS, 3)
        axis_lbl = AXIS[int(dir_align[mode_idx])]
        print(f"\n-- Mode {mode_idx:2d} | {freqs[mode_idx]:>12.2f} Hz | aligns with {axis_lbl}-axis (dir {int(dir_align[mode_idx])}) --")
        print(f"       {'x':>10}   {'y':>10}   {'z':>10}")
        for ion_i, row in enumerate(per_ion):
            print(f"ion {ion_i}  {row[0]:>+10.5f}   {row[1]:>+10.5f}   {row[2]:>+10.5f}")
