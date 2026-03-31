from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sim.equilibrium_seeders import PlaneDefinition


_LAB_AXES = {
    "x": np.array([1.0, 0.0, 0.0], dtype=float),
    "y": np.array([0.0, 1.0, 0.0], dtype=float),
    "z": np.array([0.0, 0.0, 1.0], dtype=float),
}


def _normalize_axis(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = np.linalg.norm(arr)
    if norm <= 0.0:
        raise ValueError("View axis must be nonzero.")
    return arr / norm


def _axis_label(vec: np.ndarray, fallback: str) -> str:
    axis = _normalize_axis(vec)
    for label, basis in _LAB_AXES.items():
        if np.allclose(axis, basis, atol=1.0e-12):
            return label
        if np.allclose(axis, -basis, atol=1.0e-12):
            return f"-{label}"
    return fallback


def _resolve_view_axes(
    plane_normal: np.ndarray | None = None,
    *,
    view_axes: tuple[np.ndarray | str, np.ndarray | str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if view_axes is not None:
        axis_u, axis_v = view_axes
        u_vec = _LAB_AXES[axis_u] if isinstance(axis_u, str) else np.asarray(axis_u, dtype=float)
        v_vec = _LAB_AXES[axis_v] if isinstance(axis_v, str) else np.asarray(axis_v, dtype=float)
        return _normalize_axis(u_vec), _normalize_axis(v_vec)

    if plane_normal is None:
        return _LAB_AXES["x"], _LAB_AXES["y"]

    plane = PlaneDefinition.from_normal(np.asarray(plane_normal, dtype=float))
    return plane.basis[:, 0], plane.basis[:, 1]


def build_equilibrium_figure(
    final_positions: np.ndarray | None,
    *,
    seed_positions: np.ndarray | None = None,
    plane_normal: np.ndarray | None = None,
    view_axes: tuple[np.ndarray | str, np.ndarray | str] | None = None,
    title: str | None = None,
):
    if final_positions is None:
        return None

    final_pts = np.asarray(final_positions, dtype=float).reshape(-1, 3)
    if final_pts.size == 0:
        return None

    axis_u, axis_v = _resolve_view_axes(plane_normal, view_axes=view_axes)
    final_proj = np.column_stack([final_pts @ axis_u, final_pts @ axis_v])

    seed_proj = None
    if seed_positions is not None:
        seed_pts = np.asarray(seed_positions, dtype=float).reshape(-1, 3)
        if seed_pts.size:
            seed_proj = np.column_stack([seed_pts @ axis_u, seed_pts @ axis_v])

    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    if seed_proj is not None:
        ax.scatter(
            seed_proj[:, 0] * 1.0e6,
            seed_proj[:, 1] * 1.0e6,
            s=18,
            c="#ff7f0e",
            alpha=0.85,
            label="winning seed",
        )
    ax.scatter(
        final_proj[:, 0] * 1.0e6,
        final_proj[:, 1] * 1.0e6,
        s=58,
        c="#1f77b4",
        edgecolors="black",
        linewidths=0.5,
        label="equilibrium",
    )
    ax.set_xlabel(f"{_axis_label(axis_u, 'u')} (um)")
    ax.set_ylabel(f"{_axis_label(axis_v, 'v')} (um)")
    ax.set_title(title or "Equilibrium positions")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    if seed_proj is not None:
        ax.legend(loc="best")
    return fig


def build_equilibrium_summary_lines(metadata: Any) -> list[str]:
    if not isinstance(metadata, dict):
        return []

    lines = []
    minimizer_name = metadata.get("minimizer_name")
    if minimizer_name:
        lines.append(f"minimizer: {minimizer_name}")

    if "success" in metadata:
        lines.append(f"success: {bool(metadata.get('success'))}")

    message = metadata.get("message")
    if message:
        lines.append(f"message: {message}")

    energy = metadata.get("energy")
    if energy is not None:
        try:
            lines.append(f"energy: {float(energy):.8g}")
        except (TypeError, ValueError):
            pass

    alpha_eff = metadata.get("alpha_eff")
    if alpha_eff is not None:
        try:
            lines.append(f"alpha: {float(alpha_eff):.6g}")
        except (TypeError, ValueError):
            pass

    seed_family = metadata.get("seed_family")
    if seed_family:
        lines.append(f"seed family: {seed_family}")

    seed_spacing = metadata.get("seed_spacing")
    if seed_spacing is not None:
        try:
            lines.append(f"seed spacing: {float(seed_spacing):.6g}")
        except (TypeError, ValueError):
            pass

    seed_jitter = metadata.get("seed_jitter")
    if seed_jitter is not None:
        try:
            lines.append(f"seed jitter: {float(seed_jitter):.6g}")
        except (TypeError, ValueError):
            pass

    seed_rng_seed = metadata.get("seed_rng_seed")
    if seed_rng_seed is not None:
        try:
            lines.append(f"seed rng seed: {int(seed_rng_seed)}")
        except (TypeError, ValueError):
            pass

    stage = metadata.get("stage")
    if stage:
        lines.append(f"winner stage: {stage}")

    trap_model = metadata.get("trap_model")
    if trap_model:
        lines.append(f"trap model: {trap_model}")

    grad_norm = metadata.get("projected_gradient_max_norm")
    if grad_norm is not None:
        try:
            lines.append(f"max projected |grad|: {float(grad_norm):.3e}")
        except (TypeError, ValueError):
            pass

    moved = metadata.get("distance_moved_from_seed_dimless")
    if moved is not None:
        try:
            lines.append(f"seed-to-final rms move: {float(moved):.5g}")
        except (TypeError, ValueError):
            pass

    if "seed_fits_bounds_directly" in metadata:
        lines.append(
            f"seed fits bounds directly: {bool(metadata.get('seed_fits_bounds_directly'))}"
        )

    elif "seed_rescaled_to_fit" in metadata:
        lines.append(f"seed rescaled to fit: {bool(metadata.get('seed_rescaled_to_fit'))}")

    fit_scale = metadata.get("seed_fit_scale")
    if fit_scale is not None:
        try:
            lines.append(f"seed fit scale: {float(fit_scale):.5g}")
        except (TypeError, ValueError):
            pass

    pair_sep = metadata.get("minimum_pair_separation")
    if pair_sep is not None:
        try:
            lines.append(f"min pair separation: {float(pair_sep) * 1.0e6:.4f} um")
        except (TypeError, ValueError):
            pass

    optimizer_name = metadata.get("optimizer_name")
    if optimizer_name:
        lines.append(f"optimizer: {optimizer_name}")
    elif metadata.get("optimizer_path"):
        lines.append(f"optimizer: {metadata.get('optimizer_path')}")

    return lines
