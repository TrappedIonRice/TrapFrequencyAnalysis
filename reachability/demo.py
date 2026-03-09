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
    plot_reachable_boundary_hull_3d,
    sample_reachable_boundary,
)


def run_reachability_demo(
    *,
    n_samples: int = 2000,
    trap_name: str = "Simp58_101",
    num_model_samples: int = 30,
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


if __name__ == "__main__":
    run_reachability_demo()
