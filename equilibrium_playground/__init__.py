"""Standalone playground for reduced 2D equilibrium experiments.

This package is intentionally independent from the main simulation and
minimizer architecture. It works directly with manually supplied reduced
polynomial coefficients in dimensionless x-z coordinates.
"""

from equilibrium_playground.cryo2d_closecopy_2 import (
    Cryo2DCloseCopy2Result,
    make_triangular_seed,
    solve_cryo2d_closecopy_2,
)
from equilibrium_playground.symmetry_enforced_harmonic2d import (
    SymmetryEnforcedHarmonic2DResult,
    make_symmetry_enforced_seed,
    solve_symmetry_enforced_harmonic2d,
)
from equilibrium_playground.viewer import (
    build_result_figure,
    format_result_summary,
)
from equilibrium_playground.wrappers import (
    BestOfKResult,
    best_of_k_cryo2d_closecopy_2,
)

__all__ = [
    "BestOfKResult",
    "Cryo2DCloseCopy2Result",
    "SymmetryEnforcedHarmonic2DResult",
    "build_result_figure",
    "best_of_k_cryo2d_closecopy_2",
    "format_result_summary",
    "make_symmetry_enforced_seed",
    "make_triangular_seed",
    "solve_cryo2d_closecopy_2",
    "solve_symmetry_enforced_harmonic2d",
]
