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
from .frequency_space import (
    DEFAULT_EDGE_SAMPLES_PER_EDGE,
    DEFAULT_FACE_SAMPLES_PER_FACE,
    DEFAULT_POSITIVE_OCTANT_TOL,
    FrequencyBoundarySample,
    PositiveOctantLambdaBoundarySample,
    convert_lambda_boundary_to_frequency_samples,
    enrich_positive_octant_lambda_boundary,
    filter_positive_octant_points,
    lambda_to_frequency_points,
    plot_frequency_boundary_points_3d,
    plot_frequency_boundary_points_3d_plotly,
    plot_multi_trap_frequency_space,
    plot_single_trap_frequency_space,
    plot_trap_frequency_space,
    positive_octant_mask,
)
from .plotting import plot_reachable_boundary_hull_3d, plot_reachable_boundary_points_3d
from .sampling import BoundarySamplingResult, sample_reachable_boundary
from .support import SupportQueryResult, solve_reachability_support_query

__all__ = [
    "ReachabilityModel",
    "ModalCurvatureHull",
    "PositiveOctantLambdaBoundarySample",
    "FrequencyBoundarySample",
    "SupportQueryResult",
    "BoundarySamplingResult",
    "DEFAULT_DEDUPLICATE_TOL",
    "DEFAULT_POSITIVE_OCTANT_TOL",
    "DEFAULT_EDGE_SAMPLES_PER_EDGE",
    "DEFAULT_FACE_SAMPLES_PER_FACE",
    "build_reachability_model",
    "build_fixed_modal_equalities_for_point",
    "build_modal_diagonal_rows_for_point",
    "build_modal_curvature_hull",
    "deduplicate_modal_curvature_points",
    "positive_octant_mask",
    "filter_positive_octant_points",
    "lambda_to_frequency_points",
    "enrich_positive_octant_lambda_boundary",
    "convert_lambda_boundary_to_frequency_samples",
    "solve_reachability_support_query",
    "sample_reachable_boundary",
    "plot_reachable_boundary_points_3d",
    "plot_reachable_boundary_hull_3d",
    "plot_frequency_boundary_points_3d",
    "plot_frequency_boundary_points_3d_plotly",
    "plot_single_trap_frequency_space",
    "plot_multi_trap_frequency_space",
    "plot_trap_frequency_space",
]
