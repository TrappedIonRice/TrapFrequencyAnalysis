"""
Dummy inverse-design runner for InnTrapFine.

This is a minimal, hard-coded example that wires the optimization together.
"""

from __future__ import annotations

import numpy as np

from inverse_design import solve_u_for_targets
from linearsyssolve101 import _build_trapping_vars_from_u
from sim.simulation import Simulation
from trapping_variables import Trapping_Vars


def run_demo() -> None:
    # Target specification (hard-coded demo values)
    r0 = np.array([0.0, 0.0, 0.0], dtype=float)
    freqs = np.array([0.3e6, 0.6e6, 3.2e6], dtype=float)  # Hz
    principal_dirs = [[1,0,0],[0,1,0],[0,0,1]]

    # Trap/fit configuration for InnTrapFine
    trap_name = "InnTrapFine"
    dc_electrodes = [f"DC{i}" for i in range(1, 13)]

    u_bounds = (
    [(-10, 10)] * 12 +    # DC1..DC12
    [(-30, 30)] * 2 +         # RF1_DC, RF2_DC
    [(0, 1000000)]             # rf2
)

    out = solve_u_for_targets(
        r0=r0,
        freqs=freqs,
        principal_dirs=principal_dirs,
        ion_mass_kg=6.64e-26,  # example: 40Ca+ in kg
        ion_charge_c=1.602176634e-19,
        poly_is_potential_energy=False,
        freqs_in_hz=True,
        trap_name=trap_name,
        dc_electrodes=dc_electrodes,
        rf_freq_hz=43e6,
        num_samples=20,
        polyfit_deg=4,
        objective="l2",
        rf2_penalty_scale=1e-6,
        enforce_bounds=True,
        u_bounds=u_bounds,
    )

    np.set_printoptions(precision=6, suppress=True)
    print("status:", out["status"])
    print("u:", out["u"])
    print("residual_norm:", out["residual_norm"])
    print("u_norm2:", out["u_norm2"])
    print("u_norminf:", out["u_norminf"])
    print("rf2:", out["rf2"])
    print("rf_amp:", out["rf_amp"])

    # Run simulation with the found u to inspect principal directions and freqs
    u = out["u"]
    tv = _build_trapping_vars_from_u(
        dc_electrodes=dc_electrodes,
        rf_dc_electrodes=["RF1", "RF2"],
        u=u,
        rf_freq_hz=43e6,
    )
    sim = Simulation(trap_name, Trapping_Vars())
    sim.change_electrode_variables(tv)
    sim.clear_held_results()

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
