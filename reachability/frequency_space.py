"""
Frequency-space utilities built on modal-curvature reachability samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import warnings
from datetime import datetime
from pathlib import Path
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import constants
from .hull import (
    DEFAULT_DEDUPLICATE_TOL,
    ModalCurvatureHull,
    build_modal_curvature_hull,
    deduplicate_modal_curvature_points,
)
from .model import ReachabilityModel, build_reachability_model
from .sampling import BoundarySamplingResult, sample_reachable_boundary

try:
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError
except Exception:  # pragma: no cover - optional dependency
    linprog = None
    ConvexHull = None
    HalfspaceIntersection = None
    QhullError = None

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency
    go = None


DEFAULT_POSITIVE_OCTANT_TOL = 1e-12
DEFAULT_EDGE_SAMPLES_PER_EDGE = 4
DEFAULT_FACE_SAMPLES_PER_FACE = 8
DEFAULT_TRAP_DC_COUNTS = {
    "InnTrapFine": 12,
    "1252dTrapRice": 20,
    "Simp58_101": 10,
}
PLOTLY_DEFAULT_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


@dataclass
class PositiveOctantLambdaBoundarySample:
    """
    Enriched boundary sample of the clipped positive-octant lambda region.
    """

    input_lambda_points: np.ndarray
    source_hull: ModalCurvatureHull | None
    clipped_vertices: np.ndarray
    clipped_hull: ConvexHull | None
    vertex_points: np.ndarray
    edge_points: np.ndarray
    face_points: np.ndarray
    surface_lambda_points: np.ndarray
    surface_triangles: np.ndarray
    lambda_points: np.ndarray
    octant_tol: float
    deduplicate_tol: float | None
    edge_samples_per_edge: int
    face_samples_per_face: int
    status: str
    message: str
    n_input_points: int
    n_clipped_vertices: int
    n_vertex_points: int
    n_edge_points: int
    n_face_points: int
    n_surface_points: int
    n_surface_triangles: int
    n_total_points: int


@dataclass
class FrequencyBoundarySample:
    """
    Frequency-space boundary sample converted from enriched lambda boundary points.
    """

    lambda_boundary: PositiveOctantLambdaBoundarySample
    frequency_points: np.ndarray
    frequency_surface_points: np.ndarray
    frequency_surface_triangles: np.ndarray
    output: str
    status: str
    message: str

    @property
    def n_points(self) -> int:
        return int(self.frequency_points.shape[0])


def positive_octant_mask(
    lambda_points: np.ndarray,
    *,
    tol: float = DEFAULT_POSITIVE_OCTANT_TOL,
) -> np.ndarray:
    """
    Mask points in the positive octant, allowing a small negative tolerance.
    """
    pts = _as_points(lambda_points)
    if tol < 0.0:
        raise ValueError("tol must be >= 0")
    return np.all(pts >= -float(tol), axis=1)


def filter_positive_octant_points(
    lambda_points: np.ndarray,
    *,
    tol: float = DEFAULT_POSITIVE_OCTANT_TOL,
    deduplicate_tol: float | None = None,
) -> np.ndarray:
    """
    Keep positive-octant lambda points and optionally deduplicate them.
    """
    pts = _as_points(lambda_points)
    mask = positive_octant_mask(pts, tol=tol)
    kept = pts[mask].copy()
    kept[np.abs(kept) <= tol] = 0.0
    if deduplicate_tol is not None and kept.shape[0] > 0:
        kept, _ = deduplicate_modal_curvature_points(kept, tol=deduplicate_tol)
    return kept


def lambda_to_frequency_points(
    lambda_points: np.ndarray,
    model: ReachabilityModel,
    *,
    output: str = "hz",
    positive_octant_tol: float = DEFAULT_POSITIVE_OCTANT_TOL,
) -> np.ndarray:
    """
    Convert modal-curvature points to mode-frequency points.

    Conversion follows model conventions:
      - if model.basis == 'nondim': lambda_phys = lambda_nondim / nd_L0_m^2
      - if poly_is_potential_energy:
            lambda_phys = m * omega^2
        else:
            lambda_phys = (m/q) * omega^2
    """
    pts = _as_points(lambda_points)
    out_mode = str(output).strip().lower()
    if out_mode not in ("hz", "omega"):
        raise ValueError("output must be 'hz' or 'omega'")
    if positive_octant_tol < 0.0:
        raise ValueError("positive_octant_tol must be >= 0")

    lam_phys = _lambda_to_physical_curvature(pts, model)
    if np.any(lam_phys < -positive_octant_tol):
        raise ValueError("lambda points include values below the positive-octant tolerance")
    lam_nonneg = np.maximum(lam_phys, 0.0)

    mass = float(model.ion_mass_kg)
    if mass <= 0.0:
        raise ValueError("model.ion_mass_kg must be positive for frequency conversion")

    if bool(model.poly_is_potential_energy):
        omega2 = lam_nonneg / mass
    else:
        charge = model.ion_charge_c
        if charge is None:
            raise ValueError("model.ion_charge_c is required when poly_is_potential_energy is False")
        omega2 = lam_nonneg * (float(charge) / mass)

    if np.any(omega2 < -positive_octant_tol):
        raise ValueError("computed omega^2 has negative values outside tolerance")
    omega = np.sqrt(np.maximum(omega2, 0.0))
    if out_mode == "omega":
        return omega
    return omega / (2.0 * np.pi)


def enrich_positive_octant_lambda_boundary(
    lambda_points: np.ndarray,
    *,
    hull_result: ModalCurvatureHull | None = None,
    octant_tol: float = DEFAULT_POSITIVE_OCTANT_TOL,
    deduplicate_tol: float | None = DEFAULT_DEDUPLICATE_TOL,
    edge_samples_per_edge: int = DEFAULT_EDGE_SAMPLES_PER_EDGE,
    face_samples_per_face: int = DEFAULT_FACE_SAMPLES_PER_FACE,
    random_seed: int | None = None,
) -> PositiveOctantLambdaBoundarySample:
    """
    Build a loose enriched boundary sample for the clipped positive-octant region.

    For full 3D hulls, this clips the lambda hull to the +++ octant and samples:
      - clipped vertices,
      - points along clipped edges,
      - deterministic structured points across clipped faces.
    This naturally includes clipping-induced faces on coordinate planes.
    """
    pts = _as_points(lambda_points)
    if octant_tol < 0.0:
        raise ValueError("octant_tol must be >= 0")
    if deduplicate_tol is not None and deduplicate_tol <= 0.0:
        raise ValueError("deduplicate_tol must be positive when provided")
    if edge_samples_per_edge < 0 or face_samples_per_face < 0:
        raise ValueError("edge_samples_per_edge and face_samples_per_face must be >= 0")

    source = hull_result
    if source is None:
        source = build_modal_curvature_hull(pts, deduplicate_tol=deduplicate_tol)

    if source.hull is None:
        positive = filter_positive_octant_points(
            source.points,
            tol=octant_tol,
            deduplicate_tol=deduplicate_tol,
        )
        return PositiveOctantLambdaBoundarySample(
            input_lambda_points=pts,
            source_hull=source,
            clipped_vertices=positive,
            clipped_hull=None,
            vertex_points=positive,
            edge_points=np.zeros((0, 3), dtype=float),
            face_points=np.zeros((0, 3), dtype=float),
            surface_lambda_points=np.zeros((0, 3), dtype=float),
            surface_triangles=np.zeros((0, 3), dtype=int),
            lambda_points=positive,
            octant_tol=float(octant_tol),
            deduplicate_tol=deduplicate_tol,
            edge_samples_per_edge=int(edge_samples_per_edge),
            face_samples_per_face=int(face_samples_per_face),
            status="point_cloud_only",
            message="Source hull unavailable; used positive-octant point filtering only.",
            n_input_points=int(pts.shape[0]),
            n_clipped_vertices=int(positive.shape[0]),
            n_vertex_points=int(positive.shape[0]),
            n_edge_points=0,
            n_face_points=0,
            n_surface_points=0,
            n_surface_triangles=0,
            n_total_points=int(positive.shape[0]),
        )

    clipped_vertices, clipped_hull, clip_status, clip_message = _clip_hull_to_positive_octant(
        source.hull,
        octant_tol=octant_tol,
        deduplicate_tol=deduplicate_tol,
    )
    if clipped_vertices.shape[0] == 0:
        return PositiveOctantLambdaBoundarySample(
            input_lambda_points=pts,
            source_hull=source,
            clipped_vertices=clipped_vertices,
            clipped_hull=clipped_hull,
            vertex_points=clipped_vertices,
            edge_points=np.zeros((0, 3), dtype=float),
            face_points=np.zeros((0, 3), dtype=float),
            surface_lambda_points=np.zeros((0, 3), dtype=float),
            surface_triangles=np.zeros((0, 3), dtype=int),
            lambda_points=clipped_vertices,
            octant_tol=float(octant_tol),
            deduplicate_tol=deduplicate_tol,
            edge_samples_per_edge=int(edge_samples_per_edge),
            face_samples_per_face=int(face_samples_per_face),
            status=clip_status,
            message=clip_message,
            n_input_points=int(pts.shape[0]),
            n_clipped_vertices=0,
            n_vertex_points=0,
            n_edge_points=0,
            n_face_points=0,
            n_surface_points=0,
            n_surface_triangles=0,
            n_total_points=0,
        )

    vertex_points = clipped_vertices.copy()
    edge_points = np.zeros((0, 3), dtype=float)
    face_points = np.zeros((0, 3), dtype=float)
    surface_lambda_points = np.zeros((0, 3), dtype=float)
    surface_triangles = np.zeros((0, 3), dtype=int)
    if clipped_hull is not None:
        edge_points = _sample_hull_edges(
            clipped_vertices,
            clipped_hull.simplices,
            n_per_edge=edge_samples_per_edge,
        )
        surface_lambda_points, surface_triangles = _sample_hull_faces_structured(
            clipped_vertices,
            clipped_hull.simplices,
            subdivisions=face_samples_per_face,
        )
        face_points = surface_lambda_points

    all_points = np.vstack([vertex_points, edge_points, face_points])
    if deduplicate_tol is not None and all_points.shape[0] > 0:
        all_points, _ = deduplicate_modal_curvature_points(all_points, tol=deduplicate_tol)
    tol_keep = float(octant_tol)
    if all_points.shape[0] > 0:
        tol_keep = tol_keep + 1000.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(all_points))))
    all_points[np.abs(all_points) <= tol_keep] = 0.0
    all_points = all_points[np.all(all_points >= -tol_keep, axis=1)]
    if surface_lambda_points.shape[0] > 0:
        surface_lambda_points = surface_lambda_points.copy()
        surface_lambda_points[np.abs(surface_lambda_points) <= tol_keep] = 0.0
        keep_surface = np.all(surface_lambda_points >= -tol_keep, axis=1)
        if not np.all(keep_surface):
            # Surface points are generated from clipped hull facets, so this should
            # be rare; if it occurs, drop invalid points and disable triangles.
            surface_lambda_points = surface_lambda_points[keep_surface]
            surface_triangles = np.zeros((0, 3), dtype=int)

    status = "ok" if clipped_hull is not None else "clipped_vertices_only"
    message = clip_message
    if clipped_hull is None:
        message = (
            f"{clip_message} Clipped vertices are available but the clipped hull "
            "is degenerate; edge/face sampling was skipped."
        )

    return PositiveOctantLambdaBoundarySample(
        input_lambda_points=pts,
        source_hull=source,
        clipped_vertices=clipped_vertices,
        clipped_hull=clipped_hull,
        vertex_points=vertex_points,
        edge_points=edge_points,
        face_points=face_points,
        surface_lambda_points=surface_lambda_points,
        surface_triangles=surface_triangles,
        lambda_points=all_points,
        octant_tol=float(octant_tol),
        deduplicate_tol=deduplicate_tol,
        edge_samples_per_edge=int(edge_samples_per_edge),
        face_samples_per_face=int(face_samples_per_face),
        status=status,
        message=message,
        n_input_points=int(pts.shape[0]),
        n_clipped_vertices=int(clipped_vertices.shape[0]),
        n_vertex_points=int(vertex_points.shape[0]),
        n_edge_points=int(edge_points.shape[0]),
        n_face_points=int(face_points.shape[0]),
        n_surface_points=int(surface_lambda_points.shape[0]),
        n_surface_triangles=int(surface_triangles.shape[0]),
        n_total_points=int(all_points.shape[0]),
    )


def convert_lambda_boundary_to_frequency_samples(
    model: ReachabilityModel,
    boundary_sampling: BoundarySamplingResult,
    *,
    output: str = "hz",
    octant_tol: float = DEFAULT_POSITIVE_OCTANT_TOL,
    deduplicate_tol: float | None = DEFAULT_DEDUPLICATE_TOL,
    edge_samples_per_edge: int | None = None,
    face_samples_per_face: int | None = None,
    density_scale: float = 1.5,
    random_seed: int | None = None,
) -> FrequencyBoundarySample:
    """
    Convert sampled lambda-space boundary data into enriched frequency-space samples.

    Surface points/triangles are mapped from lambda-space face samples while
    preserving lambda-space connectivity for rendering.
    """
    edge_count = _resolve_density_count(
        edge_samples_per_edge,
        DEFAULT_EDGE_SAMPLES_PER_EDGE,
        density_scale,
    )
    face_count = _resolve_density_count(
        face_samples_per_face,
        DEFAULT_FACE_SAMPLES_PER_FACE,
        density_scale,
    )
    lambda_boundary = enrich_positive_octant_lambda_boundary(
        boundary_sampling.lambda_points,
        hull_result=boundary_sampling.hull,
        octant_tol=octant_tol,
        deduplicate_tol=deduplicate_tol,
        edge_samples_per_edge=edge_count,
        face_samples_per_face=face_count,
        random_seed=random_seed,
    )
    if lambda_boundary.lambda_points.shape[0] == 0:
        return FrequencyBoundarySample(
            lambda_boundary=lambda_boundary,
            frequency_points=np.zeros((0, 3), dtype=float),
            frequency_surface_points=np.zeros((0, 3), dtype=float),
            frequency_surface_triangles=np.zeros((0, 3), dtype=int),
            output=output,
            status="empty",
            message="No positive-octant lambda boundary points were available for conversion.",
        )

    freq_points = lambda_to_frequency_points(
        lambda_boundary.lambda_points,
        model,
        output=output,
        positive_octant_tol=octant_tol,
    )
    freq_surface_points = np.zeros((0, 3), dtype=float)
    freq_surface_triangles = np.zeros((0, 3), dtype=int)
    if lambda_boundary.surface_lambda_points.shape[0] > 0 and lambda_boundary.surface_triangles.shape[0] > 0:
        freq_surface_points = lambda_to_frequency_points(
            lambda_boundary.surface_lambda_points,
            model,
            output=output,
            positive_octant_tol=octant_tol,
        )
        freq_surface_triangles = np.asarray(lambda_boundary.surface_triangles, dtype=int)

    return FrequencyBoundarySample(
        lambda_boundary=lambda_boundary,
        frequency_points=freq_points,
        frequency_surface_points=freq_surface_points,
        frequency_surface_triangles=freq_surface_triangles,
        output=str(output).strip().lower(),
        status="ok",
        message="Frequency-space conversion completed.",
    )


def plot_frequency_boundary_points_3d(
    frequency_points: np.ndarray,
    *,
    output: str = "hz",
    show_surface: bool = True,
    surface_alpha: float = 0.18,
    ax=None,
    show: bool = True,
    label: str | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot frequency-space boundary points as a 3D scatter with optional rough shell.

    The shell is a visualization aid only: it is built from a convex hull of the
    sampled frequency points and is not an exact analytical boundary reconstruction.
    """
    pts = _as_points(frequency_points)
    mode = _validate_output_mode(output)

    if ax is None:
        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    plot_pts, x_label, y_label, z_label, default_title = _plot_display_points_and_labels(
        pts,
        mode,
    )

    if (
        show_surface
        and plot_pts.shape[0] >= 4
        and ConvexHull is not None
    ):
        try:
            shell = ConvexHull(plot_pts)
            tri = [plot_pts[simplex] for simplex in np.asarray(shell.simplices, dtype=int)]
            poly = Poly3DCollection(
                tri,
                facecolors="C1",
                edgecolors="k",
                linewidths=0.3,
                alpha=float(surface_alpha),
            )
            ax.add_collection3d(poly)
        except Exception:
            # Graceful fallback: keep scatter only when shell construction fails.
            pass

    if pts.shape[0] > 0:
        ax.scatter(
            plot_pts[:, 0],
            plot_pts[:, 1],
            plot_pts[:, 2],
            s=14,
            alpha=0.82,
            label=label,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(default_title if title is None else title)

    if label is not None:
        ax.legend(loc="best")
    if show:
        plt.show()
    return fig, ax


def plot_frequency_boundary_points_3d_plotly(
    frequency_points: np.ndarray,
    *,
    surface_points: np.ndarray | None = None,
    surface_triangles: np.ndarray | None = None,
    output: str = "hz",
    show_surface: bool = True,
    surface_alpha: float = 0.18,
    marker_size: float = 3.5,
    marker_opacity: float = 0.86,
    max_surface_triangles: int | None = 12000,
    max_scatter_points: int | None = 40000,
    fig=None,
    show: bool = True,
    label: str | None = None,
    title: str | None = None,
    color: str | None = None,
):
    """
    Plot frequency-space boundary points with Plotly 3D interactivity.

    The optional translucent shell is a visualization aid only. When provided,
    shell triangles are expected to come from mapped lambda-space boundary-face
    connectivity (not from a fresh frequency-space hull reconstruction).
    """
    if go is None:
        raise RuntimeError("Plotly is unavailable; install plotly or use matplotlib backend.")

    pts = _as_points(frequency_points)
    mode = _validate_output_mode(output)
    plot_pts, x_label, y_label, z_label, default_title = _plot_display_points_and_labels(
        pts,
        mode,
    )

    if fig is None:
        fig = go.Figure()

    surf_pts = np.zeros((0, 3), dtype=float) if surface_points is None else _as_points(surface_points)
    surf_tri = np.zeros((0, 3), dtype=int) if surface_triangles is None else _as_triangles(surface_triangles)
    if show_surface and surf_pts.shape[0] > 0 and surf_tri.shape[0] > 0:
        surf_plot_pts, _, _, _, _ = _plot_display_points_and_labels(surf_pts, mode)
        valid_tri = np.all((surf_tri >= 0) & (surf_tri < surf_plot_pts.shape[0]), axis=1)
        surf_tri_valid = surf_tri[valid_tri]
        surf_plot_pts, surf_tri_valid = _decimate_surface_mesh(
            surf_plot_pts,
            surf_tri_valid,
            max_triangles=max_surface_triangles,
        )
        if surf_tri_valid.shape[0] > 0:
            fig.add_trace(
                go.Mesh3d(
                    x=surf_plot_pts[:, 0],
                    y=surf_plot_pts[:, 1],
                    z=surf_plot_pts[:, 2],
                    i=surf_tri_valid[:, 0],
                    j=surf_tri_valid[:, 1],
                    k=surf_tri_valid[:, 2],
                    opacity=float(surface_alpha),
                    color=color or "#888888",
                    hoverinfo="skip",
                    showlegend=False,
                    name=None if label is None else f"{label} shell",
                )
            )

    scatter_pts = _decimate_scatter_points(plot_pts, max_points=max_scatter_points)
    if scatter_pts.shape[0] > 0:
        marker = {
            "size": float(marker_size),
            "opacity": float(marker_opacity),
        }
        if color is not None:
            marker["color"] = color
        fig.add_trace(
            go.Scatter3d(
                x=scatter_pts[:, 0],
                y=scatter_pts[:, 1],
                z=scatter_pts[:, 2],
                mode="markers",
                name=label,
                legendgroup=label,
                showlegend=label is not None,
                marker=marker,
            )
        )

    _update_plotly_frequency_layout(
        fig,
        x_label=x_label,
        y_label=y_label,
        z_label=z_label,
        title=default_title if title is None else title,
    )
    if show:
        fig.show()
    return fig


def plot_lambda_boundary_points_3d_plotly(
    lambda_points: np.ndarray,
    *,
    surface_points: np.ndarray | None = None,
    surface_triangles: np.ndarray | None = None,
    show_surface: bool = True,
    surface_alpha: float = 0.18,
    marker_size: float = 3.5,
    marker_opacity: float = 0.86,
    max_surface_triangles: int | None = 12000,
    max_scatter_points: int | None = 40000,
    fig=None,
    show: bool = True,
    label: str | None = None,
    title: str | None = None,
    color: str | None = None,
):
    """
    Plot lambda-space boundary points with Plotly 3D interactivity.

    The lambda-space shell uses the already-clipped positive-octant boundary
    triangles in nondimensional lambda coordinates.
    """
    if go is None:
        raise RuntimeError("Plotly is unavailable; install plotly or use matplotlib backend.")

    pts = _as_points(lambda_points)
    if fig is None:
        fig = go.Figure()

    surf_pts = np.zeros((0, 3), dtype=float) if surface_points is None else _as_points(surface_points)
    surf_tri = np.zeros((0, 3), dtype=int) if surface_triangles is None else _as_triangles(surface_triangles)
    if show_surface and surf_pts.shape[0] > 0 and surf_tri.shape[0] > 0:
        valid_tri = np.all((surf_tri >= 0) & (surf_tri < surf_pts.shape[0]), axis=1)
        surf_tri_valid = surf_tri[valid_tri]
        surf_pts_plot, surf_tri_valid = _decimate_surface_mesh(
            surf_pts,
            surf_tri_valid,
            max_triangles=max_surface_triangles,
        )
        if surf_tri_valid.shape[0] > 0:
            fig.add_trace(
                go.Mesh3d(
                    x=surf_pts_plot[:, 0],
                    y=surf_pts_plot[:, 1],
                    z=surf_pts_plot[:, 2],
                    i=surf_tri_valid[:, 0],
                    j=surf_tri_valid[:, 1],
                    k=surf_tri_valid[:, 2],
                    opacity=float(surface_alpha),
                    color=color or "#888888",
                    hoverinfo="skip",
                    showlegend=False,
                    name=None if label is None else f"{label} shell",
                )
            )

    scatter_pts = _decimate_scatter_points(pts, max_points=max_scatter_points)
    if scatter_pts.shape[0] > 0:
        marker = {
            "size": float(marker_size),
            "opacity": float(marker_opacity),
        }
        if color is not None:
            marker["color"] = color
        fig.add_trace(
            go.Scatter3d(
                x=scatter_pts[:, 0],
                y=scatter_pts[:, 1],
                z=scatter_pts[:, 2],
                mode="markers",
                name=label,
                legendgroup=label,
                showlegend=label is not None,
                marker=marker,
            )
        )

    _update_plotly_lambda_layout(
        fig,
        title="Reachable Boundary in Lambda Space" if title is None else title,
    )
    if show:
        fig.show()
    return fig


def plot_single_trap_frequency_space(
    model: ReachabilityModel,
    boundary_sampling: BoundarySamplingResult,
    *,
    output: str = "hz",
    octant_tol: float = DEFAULT_POSITIVE_OCTANT_TOL,
    deduplicate_tol: float | None = DEFAULT_DEDUPLICATE_TOL,
    edge_samples_per_edge: int | None = None,
    face_samples_per_face: int | None = None,
    density_scale: float = 1.5,
    show_surface: bool = True,
    max_surface_triangles: int | None = 12000,
    max_scatter_points: int | None = 40000,
    backend: str = "plotly",
    random_seed: int | None = None,
    ax=None,
    show: bool = True,
    label: str | None = None,
) -> tuple[Any, Any | None, FrequencyBoundarySample]:
    """
    Plot one trap's reachability boundary in frequency space.

    backend:
      - "plotly" (default): interactive 3D figure with cube aspect mode.
      - "matplotlib": existing static fallback path.
    """
    backend_mode = _resolve_plot_backend(backend)
    freq_sample = convert_lambda_boundary_to_frequency_samples(
        model,
        boundary_sampling,
        output=output,
        octant_tol=octant_tol,
        deduplicate_tol=deduplicate_tol,
        edge_samples_per_edge=edge_samples_per_edge,
        face_samples_per_face=face_samples_per_face,
        density_scale=density_scale,
        random_seed=random_seed,
    )
    if backend_mode == "plotly":
        fig = plot_frequency_boundary_points_3d_plotly(
            freq_sample.frequency_points,
            surface_points=freq_sample.frequency_surface_points,
            surface_triangles=freq_sample.frequency_surface_triangles,
            output=output,
            show_surface=show_surface,
            max_surface_triangles=max_surface_triangles,
            max_scatter_points=max_scatter_points,
            show=show,
            label=label,
        )
        return fig, None, freq_sample

    fig, mpl_ax = plot_frequency_boundary_points_3d(
        freq_sample.frequency_points,
        output=output,
        show_surface=show_surface,
        ax=ax,
        show=show,
        label=label,
    )
    return fig, mpl_ax, freq_sample


def plot_multi_trap_frequency_space(
    trap_specs: Sequence[str | Mapping[str, object]],
    *,
    n_samples: int = 2000,
    num_model_samples: int = 80,
    random_seed: int = 1,
    deduplicate_tol: float | None = DEFAULT_DEDUPLICATE_TOL,
    octant_tol: float = DEFAULT_POSITIVE_OCTANT_TOL,
    edge_samples_per_edge: int | None = None,
    face_samples_per_face: int | None = None,
    density_scale: float = 1.5,
    show_surface: bool = True,
    max_surface_triangles: int | None = 9000,
    max_scatter_points: int | None = 25000,
    plot_lambda_space: bool = False,
    save_plotly_html: bool = False,
    backend: str = "plotly",
    output: str = "hz",
    show: bool = True,
) -> tuple[Any, Any | None, list[FrequencyBoundarySample]]:
    """
    Build, sample, convert, and overlay multiple traps in one frequency-space plot.

    For mapping specs, optional key `name` overrides the legend/display label.

    backend:
      - "plotly" (default): interactive overlay with one color/trace per trap.
      - "matplotlib": static fallback overlay.
    save_plotly_html:
      - if True and backend is Plotly, save an HTML view under
        `plot_multi_trap_frequency_space_htmlViews/` in the repo root.
        If `plot_lambda_space=True`, save a second lambda-space HTML under
        `plot_multi_trap_lambda_space_htmlViews/`.
    """
    backend_mode = _resolve_plot_backend(backend)
    mode = _validate_output_mode(output)
    default_title = (
        "Reachable Boundary in Frequency Space (MHz)"
        if mode == "hz"
        else "Reachable Boundary in Frequency Space (omega)"
    )

    if backend_mode == "plotly":
        fig = go.Figure()
        lambda_fig = go.Figure() if plot_lambda_space else None
        results: list[FrequencyBoundarySample] = []
        labels_for_name: list[str] = []
        for idx, spec in enumerate(trap_specs):
            print(f"[info] Processing trap {idx + 1}/{len(trap_specs)}: {spec}")
            cfg = _normalize_trap_spec(spec)
            trap_label = str(cfg.get("name", cfg["trap_name"]))
            labels_for_name.append(trap_label)
            model = _build_model_from_spec(cfg, num_model_samples=num_model_samples)
            boundary = sample_reachable_boundary(
                model,
                n_samples=n_samples,
                random_seed=random_seed + idx,
                deduplicate_tol=deduplicate_tol,
                build_hull=True,
            )
            freq_sample = convert_lambda_boundary_to_frequency_samples(
                model,
                boundary,
                output=output,
                octant_tol=octant_tol,
                deduplicate_tol=deduplicate_tol,
                edge_samples_per_edge=edge_samples_per_edge,
                face_samples_per_face=face_samples_per_face,
                density_scale=density_scale,
                random_seed=random_seed + idx,
            )
            results.append(freq_sample)
            color = PLOTLY_DEFAULT_COLORS[idx % len(PLOTLY_DEFAULT_COLORS)]
            fig = plot_frequency_boundary_points_3d_plotly(
                freq_sample.frequency_points,
                surface_points=freq_sample.frequency_surface_points,
                surface_triangles=freq_sample.frequency_surface_triangles,
                output=output,
                show_surface=show_surface,
                max_surface_triangles=max_surface_triangles,
                max_scatter_points=max_scatter_points,
                show=False,
                label=str(trap_label),
                color=color,
                fig=fig,
                title=default_title,
            )
            if lambda_fig is not None:
                lambda_fig = plot_lambda_boundary_points_3d_plotly(
                    freq_sample.lambda_boundary.lambda_points,
                    surface_points=freq_sample.lambda_boundary.surface_lambda_points,
                    surface_triangles=freq_sample.lambda_boundary.surface_triangles,
                    show_surface=show_surface,
                    max_surface_triangles=max_surface_triangles,
                    max_scatter_points=max_scatter_points,
                    show=False,
                    label=str(trap_label),
                    color=color,
                    fig=lambda_fig,
                )
        if save_plotly_html:
            out_dir = _default_multi_trap_html_output_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / _build_multi_trap_html_filename(
                labels=labels_for_name,
                n_samples=n_samples,
                output=output,
                density_scale=density_scale,
                show_surface=show_surface,
            )
            try:
                fig.write_html(str(out_path))
                print(f"[info] Saved multi-trap Plotly HTML: {out_path}")
            except (FileNotFoundError, OSError):
                fallback_name = _build_multi_trap_html_fallback_filename(labels=labels_for_name)
                fallback_path = out_dir / fallback_name
                fig.write_html(str(fallback_path))
                print(
                    "[warn] Primary multi-trap HTML filename was too long for this system; "
                    f"saved using compact fallback: {fallback_path}"
                )
            if lambda_fig is not None:
                lambda_out_dir = _default_multi_trap_lambda_html_output_dir()
                lambda_out_dir.mkdir(parents=True, exist_ok=True)
                lambda_out_path = lambda_out_dir / _build_multi_trap_lambda_html_filename(
                    labels=labels_for_name,
                    n_samples=n_samples,
                    density_scale=density_scale,
                    show_surface=show_surface,
                )
                try:
                    lambda_fig.write_html(str(lambda_out_path))
                    print(f"[info] Saved multi-trap lambda Plotly HTML: {lambda_out_path}")
                except (FileNotFoundError, OSError):
                    lambda_fallback_name = _build_multi_trap_lambda_html_fallback_filename(
                        labels=labels_for_name,
                    )
                    lambda_fallback_path = lambda_out_dir / lambda_fallback_name
                    lambda_fig.write_html(str(lambda_fallback_path))
                    print(
                        "[warn] Primary multi-trap lambda HTML filename was too long for this system; "
                        f"saved using compact fallback: {lambda_fallback_path}"
                    )
        if show:
            fig.show()
            if lambda_fig is not None:
                lambda_fig.show()
        return fig, None, results

    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    results: list[FrequencyBoundarySample] = []
    print("hi")
    for idx, spec in enumerate(trap_specs):
        print(f"[info] Processing trap {idx + 1}/{len(trap_specs)}: {spec}")
        cfg = _normalize_trap_spec(spec)
        trap_label = str(cfg.get("name", cfg["trap_name"]))
        model = _build_model_from_spec(cfg, num_model_samples=num_model_samples)
        boundary = sample_reachable_boundary(
            model,
            n_samples=n_samples,
            random_seed=random_seed + idx,
            deduplicate_tol=deduplicate_tol,
            build_hull=True,
        )
        _, _, freq_sample = plot_single_trap_frequency_space(
            model,
            boundary,
            output=output,
            octant_tol=octant_tol,
            deduplicate_tol=deduplicate_tol,
            edge_samples_per_edge=edge_samples_per_edge,
            face_samples_per_face=face_samples_per_face,
            density_scale=density_scale,
            show_surface=show_surface,
            max_surface_triangles=max_surface_triangles,
            max_scatter_points=max_scatter_points,
            backend=backend_mode,
            random_seed=random_seed + idx,
            ax=ax,
            show=False,
            label=trap_label,
        )
        results.append(freq_sample)
    if show:
        plt.show()
    return fig, ax, results


def plot_trap_frequency_space(
    traps: str | Mapping[str, object] | Sequence[str | Mapping[str, object]],
    *,
    n_samples: int = 2000,
    num_model_samples: int = 80,
    random_seed: int = 1,
    deduplicate_tol: float | None = DEFAULT_DEDUPLICATE_TOL,
    octant_tol: float = DEFAULT_POSITIVE_OCTANT_TOL,
    edge_samples_per_edge: int | None = None,
    face_samples_per_face: int | None = None,
    density_scale: float = 1.5,
    show_surface: bool = True,
    max_surface_triangles: int | None = 12000,
    max_scatter_points: int | None = 40000,
    plot_lambda_space: bool = False,
    backend: str = "plotly",
    output: str = "hz",
    show: bool = True,
):
    """
    Dispatch frequency-space plotting for one trap or multiple traps.

    backend defaults to "plotly" and falls back to "matplotlib" if Plotly is not
    installed in the current environment. If `plot_lambda_space=True` and Plotly
    is active, a separate lambda-space figure is also produced from the already
    clipped positive-octant lambda boundary.
    """
    backend_mode = _resolve_plot_backend(backend)
    if isinstance(traps, (str, Mapping)):
        cfg = _normalize_trap_spec(traps)
        model = _build_model_from_spec(cfg, num_model_samples=num_model_samples)
        boundary = sample_reachable_boundary(
            model,
            n_samples=n_samples,
            random_seed=random_seed,
            deduplicate_tol=deduplicate_tol,
            build_hull=True,
        )
        out = plot_single_trap_frequency_space(
            model,
            boundary,
            output=output,
            octant_tol=octant_tol,
            deduplicate_tol=deduplicate_tol,
            edge_samples_per_edge=edge_samples_per_edge,
            face_samples_per_face=face_samples_per_face,
            density_scale=density_scale,
            show_surface=show_surface,
            max_surface_triangles=max_surface_triangles,
            max_scatter_points=max_scatter_points,
            backend=backend_mode,
            random_seed=random_seed,
            show=show,
            label=cfg["trap_name"],
        )
        if plot_lambda_space and backend_mode == "plotly":
            _, _, freq_sample = out
            plot_lambda_boundary_points_3d_plotly(
                freq_sample.lambda_boundary.lambda_points,
                surface_points=freq_sample.lambda_boundary.surface_lambda_points,
                surface_triangles=freq_sample.lambda_boundary.surface_triangles,
                show_surface=show_surface,
                max_surface_triangles=max_surface_triangles,
                max_scatter_points=max_scatter_points,
                show=show,
                label=cfg["trap_name"],
            )
        return out
    return plot_multi_trap_frequency_space(
        list(traps),
        n_samples=n_samples,
        num_model_samples=num_model_samples,
        random_seed=random_seed,
        deduplicate_tol=deduplicate_tol,
        octant_tol=octant_tol,
        edge_samples_per_edge=edge_samples_per_edge,
        face_samples_per_face=face_samples_per_face,
        density_scale=density_scale,
        show_surface=show_surface,
        max_surface_triangles=max_surface_triangles,
        max_scatter_points=max_scatter_points,
        plot_lambda_space=plot_lambda_space,
        backend=backend_mode,
        output=output,
        show=show,
    )


def _lambda_to_physical_curvature(
    lambda_points: np.ndarray,
    model: ReachabilityModel,
) -> np.ndarray:
    pts = _as_points(lambda_points)
    basis = str(model.basis).strip().lower()
    if basis == "nondim":
        L0 = float(model.nd_L0_m)
        if L0 <= 0.0:
            raise ValueError("model.nd_L0_m must be positive when basis is 'nondim'")
        return pts / (L0 * L0)
    if basis == "physical":
        return pts.copy()
    raise ValueError(f"Unsupported model basis '{model.basis}'.")


def _validate_output_mode(output: str) -> str:
    mode = str(output).strip().lower()
    if mode not in ("hz", "omega"):
        raise ValueError("output must be 'hz' or 'omega'")
    return mode


def _plot_display_points_and_labels(
    points: np.ndarray,
    output_mode: str,
) -> tuple[np.ndarray, str, str, str, str]:
    if output_mode == "hz":
        return (
            points / 1.0e6,
            "f_1 (MHz)",
            "f_2 (MHz)",
            "f_3 (MHz)",
            "Reachable Boundary in Frequency Space (MHz)",
        )
    return (
        points,
        "omega_1 (rad/s)",
        "omega_2 (rad/s)",
        "omega_3 (rad/s)",
        "Reachable Boundary in Frequency Space (omega)",
    )


def _update_plotly_frequency_layout(
    fig,
    *,
    x_label: str,
    y_label: str,
    z_label: str,
    title: str,
) -> None:
    fig.update_layout(
        title=title,
        scene={
            "xaxis": {"title": {"text": x_label}},
            "yaxis": {"title": {"text": y_label}},
            "zaxis": {"title": {"text": z_label}},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )


def _update_plotly_lambda_layout(
    fig,
    *,
    title: str,
) -> None:
    fig.update_layout(
        title=title,
        scene={
            "xaxis": {"title": {"text": "lambda_1"}},
            "yaxis": {"title": {"text": "lambda_2"}},
            "zaxis": {"title": {"text": "lambda_3"}},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )


def _resolve_plot_backend(backend: str) -> str:
    mode = str(backend).strip().lower()
    if mode not in ("plotly", "matplotlib"):
        raise ValueError("backend must be 'plotly' or 'matplotlib'")
    if mode == "plotly" and go is None:
        warnings.warn(
            "Plotly backend requested but plotly is unavailable; falling back to matplotlib.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "matplotlib"
    return mode


def _clip_hull_to_positive_octant(
    hull: ConvexHull,
    *,
    octant_tol: float,
    deduplicate_tol: float | None,
) -> tuple[np.ndarray, ConvexHull | None, str, str]:
    if HalfspaceIntersection is None or linprog is None or ConvexHull is None:
        return (
            np.zeros((0, 3), dtype=float),
            None,
            "scipy_unavailable",
            "SciPy half-space clipping tools are unavailable.",
        )

    halfspaces = np.asarray(hull.equations, dtype=float)
    octant_halfspaces = np.array(
        [
            [-1.0, 0.0, 0.0, -octant_tol],
            [0.0, -1.0, 0.0, -octant_tol],
            [0.0, 0.0, -1.0, -octant_tol],
        ],
        dtype=float,
    )
    all_halfspaces = np.vstack([halfspaces, octant_halfspaces])

    interior = _find_strict_interior_point(all_halfspaces)
    if interior is None:
        return (
            np.zeros((0, 3), dtype=float),
            None,
            "clip_failed",
            "Could not find a strict interior point for clipped positive-octant polytope.",
        )

    try:
        hs = HalfspaceIntersection(all_halfspaces, interior)
    except Exception as exc:  # pragma: no cover - geometry failure path
        return (
            np.zeros((0, 3), dtype=float),
            None,
            "clip_failed",
            f"HalfspaceIntersection failed: {exc}",
        )

    verts = np.asarray(hs.intersections, dtype=float)
    if verts.size == 0:
        return (
            np.zeros((0, 3), dtype=float),
            None,
            "empty",
            "Clipped positive-octant polytope has no vertices.",
        )
    verts = verts[np.all(np.isfinite(verts), axis=1)]
    tol_keep = float(octant_tol) + 1000.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(verts))))
    verts[np.abs(verts) <= tol_keep] = 0.0
    verts = verts[np.all(verts >= -tol_keep, axis=1)]
    if deduplicate_tol is not None and verts.shape[0] > 0:
        verts, _ = deduplicate_modal_curvature_points(verts, tol=deduplicate_tol)

    if verts.shape[0] < 4:
        return (
            verts,
            None,
            "insufficient_points",
            "Clipped region has fewer than 4 vertices; 3D hull is unavailable.",
        )

    try:
        clipped_hull = ConvexHull(verts)
    except QhullError as exc:
        return (
            verts,
            None,
            "degenerate",
            f"Clipped vertices are degenerate for 3D hull: {exc}",
        )
    return verts, clipped_hull, "ok", "Clipped positive-octant hull built successfully."


def _find_strict_interior_point(halfspaces: np.ndarray) -> np.ndarray | None:
    A = np.asarray(halfspaces[:, :3], dtype=float)
    b = np.asarray(halfspaces[:, 3], dtype=float)
    m = A.shape[0]

    # maximize t: A x + b + t <= 0
    c = np.array([0.0, 0.0, 0.0, -1.0], dtype=float)
    A_ub = np.hstack([A, np.ones((m, 1), dtype=float)])
    b_ub = -b
    bounds = [(None, None), (None, None), (None, None), (None, None)]

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not bool(res.success) or res.x is None:
        return None
    if float(res.x[3]) <= 1e-12:
        return None
    return np.asarray(res.x[:3], dtype=float)


def _sample_hull_edges(
    vertices: np.ndarray,
    simplices: np.ndarray,
    *,
    n_per_edge: int,
) -> np.ndarray:
    if n_per_edge <= 0:
        return np.zeros((0, 3), dtype=float)
    edges = _unique_edges_from_simplices(np.asarray(simplices, dtype=int))
    if len(edges) == 0:
        return np.zeros((0, 3), dtype=float)

    tvals = np.linspace(0.0, 1.0, n_per_edge + 2, dtype=float)[1:-1]
    points = []
    for i, j in edges:
        p0 = vertices[i]
        p1 = vertices[j]
        for t in tvals:
            points.append((1.0 - t) * p0 + t * p1)
    return np.asarray(points, dtype=float) if points else np.zeros((0, 3), dtype=float)


def _sample_hull_faces_structured(
    vertices: np.ndarray,
    simplices: np.ndarray,
    *,
    subdivisions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministically sample each hull simplex with a barycentric grid.

    Returns:
      - sampled points in lambda space
      - triangle connectivity indexing into those sampled points
    """
    n = int(subdivisions)
    if n <= 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)

    points_out: list[np.ndarray] = []
    tri_out: list[np.ndarray] = []
    offset = 0
    for tri in np.asarray(simplices, dtype=int):
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        local_points, local_tri = _structured_triangle_grid(a, b, c, n)
        if local_points.shape[0] == 0 or local_tri.shape[0] == 0:
            continue
        points_out.append(local_points)
        tri_out.append(local_tri + offset)
        offset += local_points.shape[0]

    if not points_out:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)
    return np.vstack(points_out), np.vstack(tri_out).astype(int)


def _structured_triangle_grid(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    subdivisions: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(subdivisions)
    if n <= 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)

    points: list[np.ndarray] = []
    index_by_ij: dict[tuple[int, int], int] = {}
    for i in range(n + 1):
        for j in range(n + 1 - i):
            u = float(i) / float(n)
            v = float(j) / float(n)
            p = (1.0 - u - v) * a + u * b + v * c
            index_by_ij[(i, j)] = len(points)
            points.append(p)

    triangles: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(n - i):
            p0 = index_by_ij[(i, j)]
            p1 = index_by_ij[(i + 1, j)]
            p2 = index_by_ij[(i, j + 1)]
            triangles.append((p0, p1, p2))
            if j < (n - i - 1):
                p3 = index_by_ij[(i + 1, j + 1)]
                triangles.append((p1, p3, p2))

    if not points or not triangles:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)
    return np.asarray(points, dtype=float), np.asarray(triangles, dtype=int)


def _unique_edges_from_simplices(simplices: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for tri in simplices:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((i, k))))
    return sorted(edges)


def _as_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return pts


def _as_triangles(triangles: np.ndarray) -> np.ndarray:
    tri = np.asarray(triangles, dtype=int)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("triangles must have shape (M, 3)")
    return tri


def _resolve_density_count(
    explicit: int | None,
    default: int,
    scale: float,
) -> int:
    if explicit is not None:
        val = int(explicit)
        if val < 0:
            raise ValueError("density sample counts must be >= 0")
        return val
    sc = float(scale)
    if sc <= 0.0:
        raise ValueError("density_scale must be positive")
    return max(0, int(round(float(default) * sc)))


def _default_multi_trap_html_output_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "plot_multi_trap_frequency_space_htmlViews"


def _default_multi_trap_lambda_html_output_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "plot_multi_trap_lambda_space_htmlViews"


def _build_multi_trap_html_filename(
    *,
    labels: Sequence[str],
    n_samples: int,
    output: str,
    density_scale: float,
    show_surface: bool,
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_items = [_slugify_for_filename(x) for x in labels if str(x).strip()]
    full_label_slug = "__".join(label_items) if label_items else "unnamed"
    preview_items = label_items[:3]
    preview_slug = "__".join(preview_items) if preview_items else "unnamed"
    if len(label_items) > 3:
        preview_slug = f"{preview_slug}__plus-{len(label_items) - 3}"
    preview_slug_short = preview_slug[:40].strip("-_")
    if not preview_slug_short:
        preview_slug_short = "unnamed"
    descriptor = (
        f"plot_multi_trap_frequency_space__{ts}"
        f"__traps-{len(labels)}"
        f"__labels-{preview_slug_short}"
        f"__nsamples-{int(n_samples)}"
        f"__output-{str(output).strip().lower()}"
        f"__dens-{float(density_scale):.3f}"
        f"__surface-{int(bool(show_surface))}"
    )
    digest_input = (
        f"{descriptor}__full_labels={full_label_slug}"
    )
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:10]
    return f"{descriptor}__{digest}.html"


def _slugify_for_filename(value: str) -> str:
    s = "".join(ch if ch.isalnum() else "-" for ch in str(value))
    s = "-".join(part for part in s.split("-") if part)
    return s[:40] if s else "x"


def _build_multi_trap_html_fallback_filename(*, labels: Sequence[str]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1("||".join(str(x) for x in labels).encode("utf-8")).hexdigest()[:12]
    return f"plot_multi_trap_frequency_space__{ts}__{digest}.html"


def _build_multi_trap_lambda_html_filename(
    *,
    labels: Sequence[str],
    n_samples: int,
    density_scale: float,
    show_surface: bool,
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_items = [_slugify_for_filename(x) for x in labels if str(x).strip()]
    full_label_slug = "__".join(label_items) if label_items else "unnamed"
    preview_items = label_items[:3]
    preview_slug = "__".join(preview_items) if preview_items else "unnamed"
    if len(label_items) > 3:
        preview_slug = f"{preview_slug}__plus-{len(label_items) - 3}"
    preview_slug_short = preview_slug[:40].strip("-_")
    if not preview_slug_short:
        preview_slug_short = "unnamed"
    descriptor = (
        f"plot_multi_trap_lambda_space__{ts}"
        f"__traps-{len(labels)}"
        f"__labels-{preview_slug_short}"
        f"__nsamples-{int(n_samples)}"
        f"__basis-nondim"
        f"__dens-{float(density_scale):.3f}"
        f"__surface-{int(bool(show_surface))}"
    )
    digest_input = f"{descriptor}__full_labels={full_label_slug}"
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:10]
    return f"{descriptor}__{digest}.html"


def _build_multi_trap_lambda_html_fallback_filename(*, labels: Sequence[str]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1("||".join(str(x) for x in labels).encode("utf-8")).hexdigest()[:12]
    return f"plot_multi_trap_lambda_space__{ts}__{digest}.html"


def _decimate_scatter_points(points: np.ndarray, *, max_points: int | None) -> np.ndarray:
    pts = _as_points(points)
    if max_points is None:
        return pts
    cap = int(max_points)
    if cap <= 0:
        return np.zeros((0, 3), dtype=float)
    if pts.shape[0] <= cap:
        return pts
    keep = np.linspace(0, pts.shape[0] - 1, cap, dtype=int)
    return pts[keep]


def _decimate_surface_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    max_triangles: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    pts = _as_points(points)
    tri = _as_triangles(triangles)
    if max_triangles is None:
        return pts, tri
    cap = int(max_triangles)
    if cap <= 0 or tri.shape[0] == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)
    if tri.shape[0] > cap:
        keep_tri_idx = np.linspace(0, tri.shape[0] - 1, cap, dtype=int)
        tri = tri[keep_tri_idx]
    used_idx = np.unique(tri.reshape(-1))
    remap = -np.ones(pts.shape[0], dtype=int)
    remap[used_idx] = np.arange(used_idx.shape[0], dtype=int)
    tri = remap[tri]
    pts = pts[used_idx]
    return pts, tri


def _normalize_trap_spec(spec: str | Mapping[str, object]) -> dict[str, object]:
    if isinstance(spec, str):
        trap_name = spec
        dc_count = DEFAULT_TRAP_DC_COUNTS.get(trap_name)
        if dc_count is None:
            raise ValueError(
                f"No default DC-electrode count known for trap '{trap_name}'. "
                "Provide a mapping spec with explicit dc_electrodes."
            )
        return {
            "trap_name": trap_name,
            "dc_electrodes": [f"DC{i}" for i in range(1, dc_count + 1)],
            "rf_dc_electrodes": ["RF1", "RF2"],
            "r0": np.array([0.0, 0.0, 0.0], dtype=float),
            "principal_axis": np.array([1.0, 0.0, 0.0], dtype=float),
            "ref_dir": np.array([0.0, 0.0, 1.0], dtype=float),
            "alpha_deg": 0.0,
        }

    cfg = dict(spec)
    if "trap_name" not in cfg:
        raise ValueError("trap spec mapping must include 'trap_name'")
    if "dc_electrodes" not in cfg:
        trap_name = str(cfg["trap_name"])
        dc_count = DEFAULT_TRAP_DC_COUNTS.get(trap_name)
        if dc_count is None:
            raise ValueError(
                f"trap spec for '{trap_name}' must include dc_electrodes "
                "(no default count is known)."
            )
        cfg["dc_electrodes"] = [f"DC{i}" for i in range(1, dc_count + 1)]
    cfg.setdefault("rf_dc_electrodes", ["RF1", "RF2"])
    cfg.setdefault("r0", np.array([0.0, 0.0, 0.0], dtype=float))
    cfg.setdefault("principal_axis", np.array([1.0, 0.0, 0.0], dtype=float))
    cfg.setdefault("ref_dir", np.array([0.0, 0.0, 1.0], dtype=float))
    cfg.setdefault("alpha_deg", 0.0)
    return cfg


def _build_model_from_spec(
    cfg: Mapping[str, object],
    *,
    num_model_samples: int,
) -> ReachabilityModel:
    u_bounds = cfg.get("u_bounds")
    if u_bounds is None:
        n_dc = len(cfg["dc_electrodes"])
        n_rf = len(cfg["rf_dc_electrodes"])
        u_bounds = [(-100.0, 100.0)] * n_dc + [(-100.0, 100.0)] * n_rf + [(0.0, None)]

    ion_mass_cfg = cfg.get("ion_mass_kg")
    ion_mass_kg = None if ion_mass_cfg is None else float(ion_mass_cfg)
    poly_is_energy = bool(cfg.get("poly_is_potential_energy", False))
    ion_charge = cfg.get(
        "ion_charge_c",
        None if poly_is_energy else constants.ion_charge,
    )

    return build_reachability_model(
        r0=np.asarray(cfg["r0"], dtype=float),
        principal_axis=np.asarray(cfg["principal_axis"], dtype=float),
        ref_dir=np.asarray(cfg["ref_dir"], dtype=float),
        alpha_deg=float(cfg["alpha_deg"]),
        trap_name=str(cfg["trap_name"]),
        dc_electrodes=list(cfg["dc_electrodes"]),
        rf_dc_electrodes=list(cfg["rf_dc_electrodes"]),
        num_samples=int(cfg.get("num_model_samples", num_model_samples)),
        u_bounds=u_bounds,
        polyfit_deg=int(cfg.get("polyfit_deg", 4)),
        seed=int(cfg.get("seed", 0)),
        use_cache=bool(cfg.get("use_cache", True)),
        ion_mass_kg=ion_mass_kg,
        ion_charge_c=ion_charge,
        poly_is_potential_energy=poly_is_energy,
    )
