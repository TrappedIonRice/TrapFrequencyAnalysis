"""
Basic demo: build reachability model, sample boundary, and plot in 3D.
"""

from __future__ import annotations
import time

import numpy as np

import constants
from reachability import (
    DEFAULT_DEDUPLICATE_TOL,
    build_reachability_model,
    plot_multi_trap_frequency_space,
    plot_reachable_boundary_hull_3d,
    plot_trap_frequency_space,
    sample_reachable_boundary,
)


def run_reachability_demo(
    *,
    n_samples: int = 2000,
    trap_name: str = "Simp58_101",
    num_model_samples: int = 80,
    seed: int = 0,
    random_seed: int = 1,
    deduplicate_tol: float = DEFAULT_DEDUPLICATE_TOL,
) -> None:
    """
    Build one fixed modal-curvature reachability model and visualize samples.
    """
    time1 = time.time()

    r0 = np.array([0.0, 0.0, 0.0], dtype=float)
    principal_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    ref_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    alpha_deg = 45.0

    dc_electrodes = [f"DC{i}" for i in range(1, 11)]
    rf_dc_electrodes = ["RF1", "RF2"]
    u_bounds = (
        [(-100.0, 100.0)] * len(dc_electrodes)
        + [(-100.0, 100.0)] * len(rf_dc_electrodes)
        + [(0.0, constants.RF_S_MAX_DEFAULT)]
    )

    model = build_reachability_model(
        r0=r0,
        principal_axis=principal_axis,
        ref_dir=ref_dir,
        alpha_deg=alpha_deg,
        trap_name=trap_name,
        dc_electrodes=dc_electrodes,
        rf_dc_electrodes=rf_dc_electrodes,
        num_samples=num_model_samples,
        u_bounds=u_bounds,
        polyfit_deg=4,
        seed=seed,
        use_cache=True,
        ion_mass_kg=constants.ion_mass,
    )

    samples = sample_reachable_boundary(
        model,
        n_samples=n_samples,
        random_seed=random_seed,
        deduplicate_tol=deduplicate_tol,
        build_hull=True,
    )
    time2 = time.time()
    print(f"Total time: {time2 - time1:.2f} seconds")


    print("Reachability demo summary")
    print("  requested directions:", samples.n_requested)
    print("  successful LP solves:", samples.n_success)
    print("  raw boundary points:", samples.n_raw_returned)
    print("  deduplicated boundary points:", samples.n_returned)
    print("  deduplicate_tol:", samples.deduplicate_tol)
    if samples.hull is not None:
        print("  hull status:", samples.hull.status)
        print("  hull vertices:", int(samples.hull.hull_vertices.shape[0]))
    print("  cache hit for A:", model.metadata.get("cache_hit"))
    print("  cache path:", model.metadata.get("cache_path"))

    plot_reachable_boundary_hull_3d(
        samples.lambda_points,
        hull_result=samples.hull,
    )


def run_frequency_demo_single(
    *,
    n_samples: int = 1500,
    trap_name: str = "1252dTrapRice",
    random_seed: int = 1,
) -> None:
    """
    Build and plot one trap directly in frequency space (Hz).
    """
    dc_count = {
        "Simp58_101": 10,
        "InnTrapFine": 12,
        "1252dTrapRice": 20,
    }.get(trap_name, 10)
    dc_electrodes = [f"DC{i}" for i in range(1, dc_count + 1)]
    rf_dc_electrodes = ["RF1", "RF2"]
    spec = {
        "trap_name": trap_name,
        "dc_electrodes": dc_electrodes,
        "rf_dc_electrodes": rf_dc_electrodes,
        "r0": np.array([0.0, 0.0, 0.0], dtype=float),
        "principal_axis": np.array([1.0, 0.0, 0.0], dtype=float),
        "ref_dir": np.array([0.0, 0.0, 1.0], dtype=float),
        "alpha_deg": 0.0,
        "u_bounds": [(-100.0, 100.0)] * (len(dc_electrodes) + len(rf_dc_electrodes))
        + [(0.0, constants.RF_S_MAX_DEFAULT)],
        "ion_mass_kg": constants.ion_mass,
        "ion_charge_c": constants.ion_charge,
        "poly_is_potential_energy": False,
    }
    plot_trap_frequency_space(
        spec,
        n_samples=n_samples,
        random_seed=random_seed,
        output="hz",
        backend="plotly",
        density_scale=1.6,
        show_surface=True,
        show=True,
    )


def run_frequency_demo_multi(
    *,
    n_samples: int = 10000,
    random_seed: int = 2,
) -> None:
    """
    Overlay multiple trap frequency-space reachability clouds on one plot.
    """
    time1 = time.time()
    specs = [
        # {
        #     "trap_name": "NISTMOCK",
        #     "name": "NISTMOCK",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 11)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 45.0,
        #     "u_bounds": [(-10.0, 10.0)] * 12 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        {
            "trap_name": "2D_V4_3_125_blades_only_Original_trap",
            "name": "2D_V4_3_125_blades_only_Original_trap",
            "dc_electrodes": [f"DC{i}" for i in range(1, 21)],
            "rf_dc_electrodes": ["RF1", "RF2"],
            "alpha_deg": 0.0,
            "u_bounds": [(-100.0, 100.0)] * 22 + [(0.0, constants.RF_S_MAX_DEFAULT)],
            "ion_mass_kg": constants.ion_mass,
            "ion_charge_c": constants.ion_charge,
            "poly_is_potential_energy": False,
        },

        {
            "trap_name": "2D trap V4.4.125 - c - 75deg 150um ground_MORE_exposed_0.1_DC_With_RF_284_+_curv",
            "name": "2D trap V4.4.125 - c - 75deg 150um ground_MORE_exposed_0.1_DC_With_RF_284_+_curv",
            "dc_electrodes": [f"DC{i}" for i in range(1, 21)],
            "rf_dc_electrodes": ["RF1", "RF2"],
            "alpha_deg": 0.0,
            "u_bounds": [(-100.0, 100.0)] * 22 + [(0.0, constants.RF_S_MAX_DEFAULT)],
            "ion_mass_kg": constants.ion_mass,
            "ion_charge_c": constants.ion_charge,
            "poly_is_potential_energy": False,
        },
        # {
        #     "trap_name": "2Dtrap_125_45deg_200exp",
        #     "name": "2Dtrap_125_45deg_200exp___pm90bounds",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 21)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 0.0,
        #     "u_bounds": [(-90.0, 90.0)] * 22 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "2Dtrap_125_45deg_200exp",
        #     "name": "2Dtrap_125_45deg_200exp___pm75bounds",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 21)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 0.0,
        #     "u_bounds": [(-75.0, 75.0)] * 22 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "2Dtrap_125_45deg_200exp",
        #     "name": "2Dtrap_125_45deg_200exp___pm50bounds",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 21)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 0.0,
        #     "u_bounds": [(-50.0, 50.0)] * 22 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "2Dtrap_125_45deg_200exp",
        #     "name": "2Dtrap_125_45deg_200exp___pm25bounds",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 21)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 0.0,
        #     "u_bounds": [(-25.0, 25.0)] * 22 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "2Dtrap_125_45deg_200exp",
        #     "name": "2Dtrap_125_45deg_200exp___pm15bounds",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 21)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 0.0,
        #     "u_bounds": [(-15.0, 15.0)] * 22 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "2Dtrap_125_45deg_200exp",
        #     "name": "2Dtrap_125_45deg_200exp___pm10bounds",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 21)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 0.0,
        #     "u_bounds": [(-10.0, 10.0)] * 22 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "InnTrapFine",
        #     "name": "InnTrapFine",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 13)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 0.0,
        #     "u_bounds": [(-100.0, 100.0)] * 14 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "1252dTrapRice",
        #     "name": "1252dTrapRice",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 21)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 0.0,
        #     "u_bounds": [(-100.0, 100.0)] * 22 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "simp58_101",
        #     "name": "simp58_101_42deg",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 11)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 42,
        #     "u_bounds": [(-10.0, 10.0)] * 12 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "simp58_101",
        #     "name": "simp58_101_43deg",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 11)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 43,
        #     "u_bounds": [(-10.0, 10.0)] * 12 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "simp58_101",
        #     "name": "simp58_101_44deg",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 11)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 44,
        #     "u_bounds": [(-10.0, 10.0)] * 12 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "simp58_101",
        #     "name": "simp58_101_45deg",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 11)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 45,
        #     "u_bounds": [(-10.0, 10.0)] * 12 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "simp58_101",
        #     "name": "simp58_101_46deg",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 11)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 46,
        #     "u_bounds": [(-10.0, 10.0)] * 12 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "simp58_101",
        #     "name": "simp58_101_48deg",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 11)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 48,
        #     "u_bounds": [(-10.0, 10.0)] * 12 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "simp58_101",
        #     "name": "simp58_101_49deg",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 11)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 49,
        #     "u_bounds": [(-10.0, 10.0)] * 12 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
        # {
        #     "trap_name": "simp58_101",
        #     "name": "simp58_101_41deg",
        #     "dc_electrodes": [f"DC{i}" for i in range(1, 11)],
        #     "rf_dc_electrodes": ["RF1", "RF2"],
        #     "alpha_deg": 41,
        #     "u_bounds": [(-10.0, 10.0)] * 12 + [(0.0, constants.RF_S_MAX_DEFAULT)],
        #     "ion_mass_kg": constants.ion_mass,
        #     "ion_charge_c": constants.ion_charge,
        #     "poly_is_potential_energy": False,
        # },
    ]
    plot_multi_trap_frequency_space(
        specs,
        n_samples=n_samples,
        random_seed=random_seed,
        plot_lambda_space=True,
        output="hz",
        backend="plotly",
        density_scale=1.4,
        show_surface=True,
        max_surface_triangles=4000,
        max_scatter_points=10000,
        show=False,
        save_plotly_html= True,
    )
    time2 = time.time()
    print(f"Total time: {time2 - time1:.2f} seconds")

if __name__ == "__main__":
    # run_frequency_demo_single()
    run_frequency_demo_multi()
