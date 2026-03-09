"""
Reachability exploration helpers in modal-curvature space.
"""

from .model import (
    ReachabilityModel,
    build_fixed_modal_equalities_for_point,
    build_modal_diagonal_rows_for_point,
    build_reachability_model,
)
from .hull import (
    DEFAULT_DEDUPLICATE_TOL,
    ModalCurvatureHull,
    build_modal_curvature_hull,
    deduplicate_modal_curvature_points,
)
from .plotting import plot_reachable_boundary_hull_3d, plot_reachable_boundary_points_3d
from .sampling import BoundarySamplingResult, sample_reachable_boundary
from .support import SupportQueryResult, solve_reachability_support_query

__all__ = [
    "ReachabilityModel",
    "ModalCurvatureHull",
    "SupportQueryResult",
    "BoundarySamplingResult",
    "DEFAULT_DEDUPLICATE_TOL",
    "build_reachability_model",
    "build_fixed_modal_equalities_for_point",
    "build_modal_diagonal_rows_for_point",
    "build_modal_curvature_hull",
    "deduplicate_modal_curvature_points",
    "solve_reachability_support_query",
    "sample_reachable_boundary",
    "plot_reachable_boundary_points_3d",
    "plot_reachable_boundary_hull_3d",
]
