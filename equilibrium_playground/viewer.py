from __future__ import annotations

"""Standalone viewer helpers for the reduced-model playground."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from equilibrium_playground.cryo2d_closecopy_2 import Cryo2DCloseCopy2Result
from equilibrium_playground.symmetry_enforced_harmonic2d import (
    SymmetryEnforcedHarmonic2DResult,
)
from equilibrium_playground.wrappers import BestOfKResult


def build_result_figure(
    result: Cryo2DCloseCopy2Result | SymmetryEnforcedHarmonic2DResult,
    *,
    title: str | None = None,
):
    """Plot the seed in orange and the final equilibrium in blue."""

    if result is None:
        return None

    seed = np.asarray(result.seed_positions_2d, dtype=float).reshape(-1, 2)
    final = np.asarray(result.positions_2d, dtype=float).reshape(-1, 2)
    if final.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    if seed.size:
        ax.scatter(
            seed[:, 0],
            seed[:, 1],
            s=22,
            c="#ff7f0e",
            alpha=0.85,
            label="seed",
        )
    ax.scatter(
        final[:, 0],
        final[:, 1],
        s=60,
        c="#1f77b4",
        edgecolors="black",
        linewidths=0.5,
        label="equilibrium",
    )
    ax.set_xlabel("x (dimensionless)")
    ax.set_ylabel("z (dimensionless)")
    ax.set_title(title or "Reduced-model equilibrium")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    return fig


def format_result_summary(
    result: Cryo2DCloseCopy2Result | SymmetryEnforcedHarmonic2DResult,
    *,
    wrapper_result: BestOfKResult | None = None,
) -> list[str]:
    """Format a compact metadata summary for printing or display."""

    if result is None:
        return []

    if isinstance(result, SymmetryEnforcedHarmonic2DResult):
        trap = result.trap_parameters
        lines = [
            "minimizer: SymmetryEnforced_Harmonic2D",
            f"N: {int(result.total_ion_count)}",
            f"success: {bool(result.success)}",
            f"message: {result.message}",
            f"energy: {float(result.energy):.8g}",
            f"trap mode: {result.trap_input_mode}",
            f"omega_x: {float(trap['omega_x']):.6g}",
            f"omega_z: {float(trap['omega_z']):.6g}",
            f"alpha: {float(trap['alpha']):.6g}",
            f"scale: {float(trap['scale']):.6g}",
            f"a_x: {float(trap['a_x']):.6g}",
            f"a_z: {float(trap['a_z']):.6g}",
            f"num_ions_vert: {int(result.num_ions_vert)}",
            f"num_ions_hor: {int(result.num_ions_hor)}",
            f"num_ions_free: {int(result.num_ions_free)}",
            f"seed family: {result.seed_metadata['seed_family']}",
            f"seed scale: {float(result.seed_metadata['seed_scale']):.6g}",
            f"seed jitter: {float(result.seed_metadata['seed_jitter']):.6g}",
            f"seed-to-final rms move: {float(result.seed_to_final_rms_move):.6g}",
            f"max |grad_reduced|: {float(result.reduced_gradient_max_norm):.3e}",
            f"max |grad_raw|: {float(result.raw_gradient_max_norm):.3e}",
            f"optimizer: {result.optimizer_path}",
        ]
        return lines

    coeffs = result.coefficients
    lines = [
        "minimizer: Cryo2d_closecopy_2_playground",
        f"N: {len(result.positions_2d)}",
        f"success: {bool(result.success)}",
        f"message: {result.message}",
        f"energy: {float(result.energy):.8g}",
        f"a20: {float(coeffs['a20']):.6g}",
        f"a02: {float(coeffs['a02']):.6g}",
        f"a40: {float(coeffs['a40']):.6g}",
        f"a22: {float(coeffs['a22']):.6g}",
        f"a04: {float(coeffs['a04']):.6g}",
        f"alpha seed: {float(result.alpha_seed):.6g}",
        f"seed spacing: {float(result.seed_spacing):.6g}",
        f"seed jitter: {float(result.seed_jitter):.6g}",
        f"seed-to-final rms move: {float(result.seed_to_final_rms_move):.6g}",
        f"max |grad|: {float(result.projected_gradient_max_norm):.3e}",
        f"optimizer: {result.optimizer_path}",
    ]
    if wrapper_result is not None:
        lines.append(f"run count: {int(wrapper_result.run_count)}")
        lines.append(
            "energies: "
            + ", ".join(f"{float(energy):.16g}" for energy in wrapper_result.energies)
        )
    return lines


def print_result_summary(
    result: Cryo2DCloseCopy2Result | SymmetryEnforcedHarmonic2DResult,
    *,
    wrapper_result: BestOfKResult | None = None,
) -> None:
    """Print a formatted summary to stdout."""

    for line in format_result_summary(result, wrapper_result=wrapper_result):
        print(line)


def metadata_from_result(
    result: Cryo2DCloseCopy2Result | SymmetryEnforcedHarmonic2DResult,
    *,
    wrapper_result: BestOfKResult | None = None,
) -> dict[str, Any]:
    """Build a plain metadata dict for ad hoc inspection."""

    metadata = dict(result.to_metadata())
    if wrapper_result is not None:
        metadata["run_count"] = int(wrapper_result.run_count)
        metadata["energies"] = [float(energy) for energy in wrapper_result.energies]
    return metadata
