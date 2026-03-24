"""
Deduplication and convex-hull helpers in modal-curvature space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.spatial import ConvexHull, QhullError
except Exception:  # pragma: no cover - optional dependency
    ConvexHull = None
    QhullError = None


DEFAULT_DEDUPLICATE_TOL = 1e-9


@dataclass
class ModalCurvatureHull:
    """
    Convex-hull summary for modal-curvature points.
    """

    input_points: np.ndarray
    points: np.ndarray
    keep_indices: np.ndarray
    deduplicate_tol: float | None
    hull: ConvexHull | None
    hull_vertices: np.ndarray
    hull_simplices: np.ndarray
    status: str
    message: str
    n_input_points: int
    n_unique_points: int


def deduplicate_modal_curvature_points(
    lambda_points: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deduplicate nearly coincident modal-curvature points with absolute tolerance.

    Returns `(points_unique, keep_indices)` where `keep_indices` are indices into
    the original point array.
    """
    pts = _as_points(lambda_points)
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if pts.shape[0] == 0:
        return pts.copy(), np.zeros(0, dtype=int)

    keep: list[int] = []
    keep_pts: list[np.ndarray] = []
    tol2 = float(tol) ** 2
    for idx, point in enumerate(pts):
        if not keep_pts:
            keep.append(idx)
            keep_pts.append(point)
            continue
        stack = np.vstack(keep_pts)
        d2 = np.sum((stack - point) ** 2, axis=1)
        if float(np.min(d2)) > tol2:
            keep.append(idx)
            keep_pts.append(point)

    keep_idx = np.asarray(keep, dtype=int)
    return pts[keep_idx], keep_idx


def build_modal_curvature_hull(
    lambda_points: np.ndarray,
    *,
    deduplicate_tol: float | None = DEFAULT_DEDUPLICATE_TOL,
    qhull_options: str | None = None,
) -> ModalCurvatureHull:
    """
    Build a convex hull in modal-curvature space from deduplicated points.

    If there are too few or degenerate points, this returns a non-`ok` status
    with `hull=None` instead of raising.
    """
    pts = _as_points(lambda_points)
    if deduplicate_tol is not None and deduplicate_tol <= 0.0:
        raise ValueError("deduplicate_tol must be positive when provided")

    if deduplicate_tol is None:
        points_unique = pts.copy()
        keep_idx = np.arange(pts.shape[0], dtype=int)
    else:
        points_unique, keep_idx = deduplicate_modal_curvature_points(
            pts,
            tol=float(deduplicate_tol),
        )

    n_input = int(pts.shape[0])
    n_unique = int(points_unique.shape[0])

    if n_unique == 0:
        return ModalCurvatureHull(
            input_points=pts,
            points=points_unique,
            keep_indices=keep_idx,
            deduplicate_tol=deduplicate_tol,
            hull=None,
            hull_vertices=np.zeros(0, dtype=int),
            hull_simplices=np.zeros((0, 3), dtype=int),
            status="empty",
            message="No modal-curvature points were available.",
            n_input_points=n_input,
            n_unique_points=n_unique,
        )

    if ConvexHull is None:
        return ModalCurvatureHull(
            input_points=pts,
            points=points_unique,
            keep_indices=keep_idx,
            deduplicate_tol=deduplicate_tol,
            hull=None,
            hull_vertices=np.zeros(0, dtype=int),
            hull_simplices=np.zeros((0, 3), dtype=int),
            status="scipy_unavailable",
            message="SciPy spatial convex hull is unavailable.",
            n_input_points=n_input,
            n_unique_points=n_unique,
        )

    if n_unique < 4:
        return ModalCurvatureHull(
            input_points=pts,
            points=points_unique,
            keep_indices=keep_idx,
            deduplicate_tol=deduplicate_tol,
            hull=None,
            hull_vertices=np.zeros(0, dtype=int),
            hull_simplices=np.zeros((0, 3), dtype=int),
            status="insufficient_points",
            message="Need at least 4 unique points to build a 3D convex hull.",
            n_input_points=n_input,
            n_unique_points=n_unique,
        )

    try:
        hull = ConvexHull(points_unique, qhull_options=qhull_options)
    except QhullError as exc:
        return ModalCurvatureHull(
            input_points=pts,
            points=points_unique,
            keep_indices=keep_idx,
            deduplicate_tol=deduplicate_tol,
            hull=None,
            hull_vertices=np.zeros(0, dtype=int),
            hull_simplices=np.zeros((0, 3), dtype=int),
            status="degenerate",
            message=str(exc),
            n_input_points=n_input,
            n_unique_points=n_unique,
        )

    return ModalCurvatureHull(
        input_points=pts,
        points=points_unique,
        keep_indices=keep_idx,
        deduplicate_tol=deduplicate_tol,
        hull=hull,
        hull_vertices=np.asarray(hull.vertices, dtype=int),
        hull_simplices=np.asarray(hull.simplices, dtype=int),
        status="ok",
        message="Convex hull built successfully.",
        n_input_points=n_input,
        n_unique_points=n_unique,
    )


def _as_points(lambda_points: np.ndarray) -> np.ndarray:
    pts = np.asarray(lambda_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("lambda_points must have shape (N, 3)")
    return pts

