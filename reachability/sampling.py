"""
Boundary sampling helpers for modal-curvature reachability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .hull import (
    DEFAULT_DEDUPLICATE_TOL,
    ModalCurvatureHull,
    build_modal_curvature_hull,
    deduplicate_modal_curvature_points,
)
from .model import ReachabilityModel
from .support import SupportQueryResult, solve_reachability_support_query


@dataclass
class BoundarySamplingResult:
    """
    Batched support-query results from random modal-curvature directions.
    """

    sampled_directions: np.ndarray
    query_results: List[SupportQueryResult]
    success_mask: np.ndarray
    objective_values: np.ndarray
    raw_lambda_points: np.ndarray
    raw_u_points: np.ndarray | None
    lambda_points: np.ndarray
    u_points: np.ndarray | None
    n_requested: int
    n_success: int
    n_raw_returned: int
    n_returned: int
    deduplicate_tol: float | None
    hull: ModalCurvatureHull | None


def sample_reachable_boundary(
    model: ReachabilityModel,
    *,
    n_samples: int = 2000,
    random_seed: int | None = None,
    rng: np.random.Generator | None = None,
    deduplicate_tol: float | None = DEFAULT_DEDUPLICATE_TOL,
    return_controls: bool = False,
    build_hull: bool = True,
) -> BoundarySamplingResult:
    """
    Sample random directions in R^3 and solve one support query per direction.

    Directions are sampled from a normal distribution and normalized to unit
    vectors. The resulting boundary cloud lives in modal-curvature space:
    lambda = [lambda_1, lambda_2, lambda_3]^T.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if rng is not None and random_seed is not None:
        raise ValueError("provide either rng or random_seed, not both")
    if deduplicate_tol is not None and deduplicate_tol <= 0.0:
        raise ValueError("deduplicate_tol must be positive when provided")

    generator = np.random.default_rng(random_seed) if rng is None else rng
    directions = _sample_unit_directions(generator, n_samples)

    query_results: List[SupportQueryResult] = []
    success_mask = np.zeros(n_samples, dtype=bool)
    objective_values = np.full(n_samples, np.nan, dtype=float)
    lambda_list: List[np.ndarray] = []
    u_list: List[np.ndarray] | None = [] if return_controls else None

    for i in range(n_samples):
        result = solve_reachability_support_query(model, directions[i])
        query_results.append(result)
        if result.success and result.lambda_star is not None and result.u_star is not None:
            success_mask[i] = True
            objective_values[i] = result.objective_value_unit_direction
            lambda_list.append(np.asarray(result.lambda_star, dtype=float))
            if u_list is not None:
                u_list.append(np.asarray(result.u_star, dtype=float))

    if lambda_list:
        raw_lambda_points = np.vstack(lambda_list)
    else:
        raw_lambda_points = np.zeros((0, 3), dtype=float)

    if u_list is None:
        raw_u_points = None
    elif u_list:
        raw_u_points = np.vstack(u_list)
    else:
        raw_u_points = np.zeros((0, model.n_controls), dtype=float)

    hull_result: ModalCurvatureHull | None = None
    if build_hull:
        hull_result = build_modal_curvature_hull(
            raw_lambda_points,
            deduplicate_tol=deduplicate_tol,
        )
        lambda_points = hull_result.points
        if raw_u_points is None:
            u_points = None
        else:
            u_points = raw_u_points[hull_result.keep_indices]
    elif deduplicate_tol is not None:
        lambda_points, keep_idx = deduplicate_modal_curvature_points(
            raw_lambda_points,
            tol=deduplicate_tol,
        )
        if raw_u_points is None:
            u_points = None
        else:
            u_points = raw_u_points[keep_idx]
    else:
        lambda_points = raw_lambda_points
        u_points = raw_u_points

    return BoundarySamplingResult(
        sampled_directions=directions,
        query_results=query_results,
        success_mask=success_mask,
        objective_values=objective_values,
        raw_lambda_points=raw_lambda_points,
        raw_u_points=raw_u_points,
        lambda_points=lambda_points,
        u_points=u_points,
        n_requested=int(n_samples),
        n_success=int(np.sum(success_mask)),
        n_raw_returned=int(raw_lambda_points.shape[0]),
        n_returned=int(lambda_points.shape[0]),
        deduplicate_tol=deduplicate_tol,
        hull=hull_result,
    )


def _sample_unit_directions(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    vecs = rng.standard_normal((n_samples, 3))
    norms = np.linalg.norm(vecs, axis=1)
    bad = norms == 0.0
    while np.any(bad):
        vecs[bad] = rng.standard_normal((int(np.sum(bad)), 3))
        norms = np.linalg.norm(vecs, axis=1)
        bad = norms == 0.0
    return vecs / norms[:, None]

