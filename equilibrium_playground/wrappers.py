from __future__ import annotations

"""Simple repeated-run wrappers for the standalone reduced-model playground."""

from dataclasses import dataclass

from equilibrium_playground.cryo2d_closecopy_2 import (
    Cryo2DCloseCopy2Result,
    solve_cryo2d_closecopy_2,
)


@dataclass
class BestOfKResult:
    """Summary of repeated Cryo-style runs on the reduced model."""

    best_result: Cryo2DCloseCopy2Result
    all_results: list[Cryo2DCloseCopy2Result]
    energies: list[float]
    successful_energies: list[float]
    run_count: int


def best_of_k_cryo2d_closecopy_2(
    num_ions: int,
    a20: float,
    a02: float,
    a40: float,
    a22: float,
    a04: float,
    *,
    k: int = 5,
    **solver_kwargs,
) -> BestOfKResult:
    """Run the standalone solver repeatedly and keep the best result.

    Randomness is intentionally left unfixed by default, so repeated runs
    naturally explore different seeds and basin-hopping trajectories.
    """

    run_count = int(k)
    if run_count <= 0:
        raise ValueError("k must be positive")

    results: list[Cryo2DCloseCopy2Result] = []
    for _ in range(run_count):
        results.append(
            solve_cryo2d_closecopy_2(
                num_ions,
                a20,
                a02,
                a40,
                a22,
                a04,
                **solver_kwargs,
            )
        )

    successful = [res for res in results if res.success]
    ranked = successful if successful else results
    best_result = min(ranked, key=lambda res: float(res.energy))
    return BestOfKResult(
        best_result=best_result,
        all_results=results,
        energies=[float(res.energy) for res in results],
        successful_energies=[float(res.energy) for res in successful],
        run_count=run_count,
    )
