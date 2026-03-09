"""
Simple 3D plotting for sampled modal-curvature boundary points.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .hull import (
    DEFAULT_DEDUPLICATE_TOL,
    ModalCurvatureHull,
    build_modal_curvature_hull,
)


def plot_reachable_boundary_points_3d(
    lambda_points: np.ndarray,
    *,
    ax=None,
    show: bool = True,
    title: str = "Reachable Boundary in Modal-Curvature Space",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot sampled boundary points as a 3D scatter in (lambda_1, lambda_2, lambda_3).
    """
    pts = np.asarray(lambda_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("lambda_points must have shape (N, 3)")

    if ax is None:
        fig = plt.figure(figsize=(7.0, 5.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if pts.shape[0] > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, alpha=0.85)
    ax.set_xlabel("lambda_1")
    ax.set_ylabel("lambda_2")
    ax.set_zlabel("lambda_3")
    ax.set_title(title)

    if show:
        plt.show()
    return fig, ax


def plot_reachable_boundary_hull_3d(
    lambda_points: np.ndarray,
    *,
    hull_result: ModalCurvatureHull | None = None,
    deduplicate_tol: float | None = DEFAULT_DEDUPLICATE_TOL,
    ax=None,
    show: bool = True,
    title: str = "Reachable Boundary Hull in Modal-Curvature Space",
) -> Tuple[plt.Figure, plt.Axes, ModalCurvatureHull]:
    """
    Plot deduplicated boundary points and their convex hull in modal-curvature space.
    """
    if hull_result is None:
        hull_result = build_modal_curvature_hull(
            lambda_points,
            deduplicate_tol=deduplicate_tol,
        )

    pts = np.asarray(hull_result.points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("hull_result.points must have shape (N, 3)")

    if ax is None:
        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if pts.shape[0] > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=16, c="C0", alpha=0.95, label="deduplicated points")

    if hull_result.hull is not None and hull_result.hull_simplices.shape[0] > 0:
        triangles = [pts[simplex] for simplex in hull_result.hull_simplices]
        poly = Poly3DCollection(
            triangles,
            facecolors="C1",
            edgecolors="k",
            linewidths=0.5,
            alpha=0.18,
        )
        ax.add_collection3d(poly)
        ax.scatter(
            pts[hull_result.hull_vertices, 0],
            pts[hull_result.hull_vertices, 1],
            pts[hull_result.hull_vertices, 2],
            s=26,
            c="C3",
            alpha=0.95,
            label="hull vertices",
        )

    ax.set_xlabel("lambda_1")
    ax.set_ylabel("lambda_2")
    ax.set_zlabel("lambda_3")
    ax.set_title(title)
    if hull_result.hull is None:
        ax.text2D(0.02, 0.97, f"Hull status: {hull_result.status}", transform=ax.transAxes)
    if pts.shape[0] > 0:
        ax.legend(loc="best")

    if show:
        plt.show()
    return fig, ax, hull_result
