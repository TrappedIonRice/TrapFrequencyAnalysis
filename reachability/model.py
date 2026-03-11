"""
Build fixed linear reachability models in modal-curvature space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

import constants
from control_constraints import (
    build_derivative_rows,
    build_hessian_rows,
    build_modal_hessian_functional_row,
    rotation_from_axis_ref_alpha,
)
from linearsyssolve101 import build_voltage_to_c_matrix, build_voltage_to_c_matrix_cached
from trap_A_cache import DEFAULT_CACHE_DIR


@dataclass
class ReachabilityModel:
    """
    Fixed linear reachability model for one inverse-control setup.

    The model uses the same coefficient map as the inverse pipeline:
        c = A u

    Equality constraints are assembled in coefficient space:
        L_eq_c c = b_eq
    with rows enforcing:
        - gradient at r0 is zero,
        - modal off-diagonal Hessian terms are zero (h12=h13=h23=0).

    The modal diagonal-curvature map is:
        lambda = L_diag_c c
    where lambda = [lambda_1, lambda_2, lambda_3]^T and
        lambda_i = e_i^T H e_i.

    Both are pushed into control space:
        E = L_eq_c @ A
        e = b_eq
        T = L_diag_c @ A
        lambda = T u

    Feasible controls:
        P = {u : lower_u <= u <= upper_u, E u = e}

    Reachable modal-curvature set:
        Lambda = {T u : u in P}
    """

    A: np.ndarray
    powers: np.ndarray
    L_eq_c: np.ndarray
    b_eq: np.ndarray
    L_diag_c: np.ndarray
    E: np.ndarray
    e: np.ndarray
    T: np.ndarray
    lower_u: np.ndarray
    upper_u: np.ndarray
    u_bounds: List[Tuple[float | None, float | None]]
    r0: np.ndarray
    rotation: np.ndarray
    basis: str = "nondim"
    nd_L0_m: float = constants.ND_L0_M
    ion_mass_kg: float = constants.ion_mass
    ion_charge_c: float | None = constants.ion_charge
    poly_is_potential_energy: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def n_controls(self) -> int:
        return int(self.T.shape[1])

    def bounds_tuples(self) -> List[Tuple[float | None, float | None]]:
        return list(self.u_bounds)


def build_modal_diagonal_rows_for_point(
    powers: np.ndarray,
    r0: Sequence[float],
    *,
    principal_axis: Sequence[float],
    ref_dir: Sequence[float],
    alpha_deg: float,
    basis: str = "nondim",
    nd_L0_m: float = constants.ND_L0_M,
) -> np.ndarray:
    """
    Build coefficient-space rows for modal diagonal curvatures [h11, h22, h33].

    Returned matrix L_diag_c has 3 rows such that:
        L_diag_c @ c = [e1^T H e1, e2^T H e2, e3^T H e3]^T.
    """
    p = np.asarray(powers, dtype=int)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("powers must have shape (M, 3)")
    r0v = _as_vec3(r0, "r0")
    if basis not in ("nondim", "physical"):
        raise ValueError("basis must be 'nondim' or 'physical'")

    r_eval = r0v / float(nd_L0_m) if basis == "nondim" else r0v
    dxx, dyy, dzz, dxy, dxz, dyz = build_hessian_rows(p, r_eval)
    vech_rows = [dxx, dyy, dzz, dxy, dxz, dyz]

    R = rotation_from_axis_ref_alpha(principal_axis, ref_dir, alpha_deg)
    rows = [
        build_modal_hessian_functional_row(vech_rows, R[:, i], R[:, i])
        for i in range(3)
    ]
    return np.vstack(rows).astype(float)


def build_fixed_modal_equalities_for_point(
    powers: np.ndarray,
    r0: Sequence[float],
    *,
    principal_axis: Sequence[float],
    ref_dir: Sequence[float],
    alpha_deg: float,
    basis: str = "nondim",
    nd_L0_m: float = constants.ND_L0_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build fixed modal-basis equality rows used by reachability.

    Equalities are:
      - gradient at r0 equals zero,
      - modal off-diagonal Hessian terms h12=h13=h23=0 in the fixed basis.
    """
    p = np.asarray(powers, dtype=int)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("powers must have shape (M, 3)")
    r0v = _as_vec3(r0, "r0")
    if basis not in ("nondim", "physical"):
        raise ValueError("basis must be 'nondim' or 'physical'")

    r_eval = r0v / float(nd_L0_m) if basis == "nondim" else r0v
    dx, dy, dz = build_derivative_rows(p, r_eval)
    dxx, dyy, dzz, dxy, dxz, dyz = build_hessian_rows(p, r_eval)
    vech_rows = [dxx, dyy, dzz, dxy, dxz, dyz]
    R = rotation_from_axis_ref_alpha(principal_axis, ref_dir, alpha_deg)

    eq_rows: List[np.ndarray] = [dx, dy, dz]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        eq_rows.append(build_modal_hessian_functional_row(vech_rows, R[:, i], R[:, j]))
    L_eq_c = np.vstack(eq_rows).astype(float)
    b_eq = np.zeros(L_eq_c.shape[0], dtype=float)
    return L_eq_c, b_eq


def build_reachability_model(
    *,
    r0: Sequence[float],
    principal_axis: Sequence[float],
    ref_dir: Sequence[float],
    alpha_deg: float,
    trap_name: str,
    dc_electrodes: Sequence[str],
    rf_dc_electrodes: Sequence[str] = ("RF1", "RF2"),
    num_samples: int = 80,
    dc_bounds: Tuple[float, float] | Sequence[Tuple[float, float]] = (-500.0, 500.0),
    rf_dc_bounds: Tuple[float, float] | Sequence[Tuple[float, float]] = (-50.0, 50.0),
    s_bounds: Tuple[float, float] = (0.0, constants.RF_S_MAX_DEFAULT),
    u_bounds: Sequence[Tuple[float | None, float | None]] | None = None,
    polyfit_deg: int = 4,
    seed: int = 0,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_rebuild_A: bool = False,
    ion_mass_kg: float | None = None,
    ion_charge_c: float | None = constants.ion_charge,
    poly_is_potential_energy: bool | None = False,
    freqs_in_hz: bool | None = None,
) -> ReachabilityModel:
    """
    Build the fixed linear reachability model for modal-curvature exploration.

    This builder is scoped to a *fixed* setup:
      - trap geometry / electrode choice,
      - equilibrium point r0,
      - principal-axis frame (principal_axis, ref_dir, alpha_deg),
      - control bounds.

    Outputs include:
      - coefficient map A,
      - equality map (L_eq_c, b_eq) and pushed form (E, e),
      - modal diagonal-curvature map L_diag_c and pushed form T,
      - explicit lower/upper control bounds.

    Notes on parameter scope:
      - `ion_mass_kg` is validated against `constants.ion_mass` because A-building
        currently depends on simulation-side global mass handling.
      - `ion_charge_c` and `poly_is_potential_energy` do not change E/e/T
        construction, but are stored on the model for explicit lambda->frequency
        conversion utilities.
      - `freqs_in_hz` is accepted for backward compatibility and unused here.
    """
    r0v = _as_vec3(r0, "r0")
    dc_list = [str(x) for x in dc_electrodes]
    rf_list = [str(x) for x in rf_dc_electrodes]
    mass_for_A = _validated_builder_mass(ion_mass_kg)
    poly_is_energy = bool(poly_is_potential_energy)
    charge_for_conversion = _resolve_conversion_charge(
        ion_charge_c=ion_charge_c,
        poly_is_potential_energy=poly_is_energy,
    )

    if use_cache:
        out = build_voltage_to_c_matrix_cached(
            trap_name=trap_name,
            dc_electrodes=dc_list,
            rf_dc_electrodes=rf_list,
            num_samples=num_samples,
            dc_bounds=dc_bounds,
            rf_dc_bounds=rf_dc_bounds,
            s_bounds=s_bounds,
            polyfit_deg=polyfit_deg,
            seed=seed,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild_A,
            ion_mass_kg=mass_for_A,
        )
    else:
        out = build_voltage_to_c_matrix(
            trap_name=trap_name,
            dc_electrodes=dc_list,
            rf_dc_electrodes=rf_list,
            num_samples=num_samples,
            dc_bounds=dc_bounds,
            rf_dc_bounds=rf_dc_bounds,
            s_bounds=s_bounds,
            polyfit_deg=polyfit_deg,
            seed=seed,
        )

    A = np.asarray(out["A"], dtype=float)
    powers = np.asarray(out["powers"], dtype=int)
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix")

    n_expected = len(dc_list) + len(rf_list) + 1
    if A.shape[1] != n_expected:
        raise ValueError(
            f"A has {A.shape[1]} controls, but electrode layout implies {n_expected}"
        )

    # Build coefficient-space fixed equalities: grad=0 and modal off-diagonal Hessian=0.
    L_eq_c, b_eq = build_fixed_modal_equalities_for_point(
        powers,
        r0v,
        principal_axis=principal_axis,
        ref_dir=ref_dir,
        alpha_deg=alpha_deg,
        basis="nondim",
        nd_L0_m=constants.ND_L0_M,
    )

    # Build coefficient-space modal diagonal map [h11, h22, h33].
    L_diag_c = build_modal_diagonal_rows_for_point(
        powers,
        r0v,
        principal_axis=principal_axis,
        ref_dir=ref_dir,
        alpha_deg=alpha_deg,
        basis="nondim",
        nd_L0_m=constants.ND_L0_M,
    )

    E = L_eq_c @ A
    e = b_eq.copy()
    T = L_diag_c @ A

    bounds = _resolve_u_bounds(
        u_bounds=u_bounds,
        dc_bounds=dc_bounds,
        rf_dc_bounds=rf_dc_bounds,
        s_bounds=s_bounds,
        n_dc=len(dc_list),
        n_rf=len(rf_list),
        n_controls=A.shape[1],
    )
    lower_u, upper_u = _bounds_to_arrays(bounds)
    R = rotation_from_axis_ref_alpha(principal_axis, ref_dir, alpha_deg)

    metadata: Dict[str, object] = {
        "trap_name": str(trap_name),
        "dc_electrodes": dc_list,
        "rf_dc_electrodes": rf_list,
        "num_samples": int(num_samples),
        "polyfit_deg": int(polyfit_deg),
        "seed": int(seed),
        "use_cache": bool(use_cache),
        "cache_dir": str(cache_dir),
        "force_rebuild_A": bool(force_rebuild_A),
        "ion_mass_kg_requested": None if ion_mass_kg is None else float(ion_mass_kg),
        "ion_mass_kg_used": mass_for_A,
        "ion_charge_c_used_for_conversion": charge_for_conversion,
        "poly_is_potential_energy_for_conversion": poly_is_energy,
        "freqs_in_hz_reserved": freqs_in_hz,
        "api_scope_note": (
            "Model is built in modal-curvature space using fixed equalities and "
            "modal diagonal map only; frequency-conversion fields are stored for "
            "downstream conversion utilities."
        ),
        "basis": "nondim",
        "nd_L0_m": float(constants.ND_L0_M),
        "r0_m": r0v.tolist(),
        "principal_axis": _as_vec3(principal_axis, "principal_axis").tolist(),
        "ref_dir": _as_vec3(ref_dir, "ref_dir").tolist(),
        "alpha_deg": float(alpha_deg),
        "cache_hit": bool(out.get("cache_hit", False)),
        "cache_path": out.get("cache_path"),
        "cfg": out.get("cfg"),
        "u_layout": out.get("u_layout"),
    }

    return ReachabilityModel(
        A=A,
        powers=powers,
        L_eq_c=np.asarray(L_eq_c, dtype=float),
        b_eq=np.asarray(b_eq, dtype=float).reshape(-1),
        L_diag_c=np.asarray(L_diag_c, dtype=float),
        E=np.asarray(E, dtype=float),
        e=np.asarray(e, dtype=float).reshape(-1),
        T=np.asarray(T, dtype=float),
        lower_u=lower_u,
        upper_u=upper_u,
        u_bounds=bounds,
        r0=r0v,
        rotation=R,
        basis="nondim",
        nd_L0_m=float(constants.ND_L0_M),
        ion_mass_kg=mass_for_A,
        ion_charge_c=charge_for_conversion,
        poly_is_potential_energy=poly_is_energy,
        metadata=metadata,
    )


def _as_vec3(value: Sequence[float], name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=float).reshape(-1)
    if vec.shape[0] != 3:
        raise ValueError(f"{name} must have length 3")
    return vec


def _validated_builder_mass(ion_mass_kg: float | None) -> float:
    """
    Validate mass handling used by A-building.

    The current A-building path is tied to simulation-side global constants. To
    avoid a misleading API, this builder rejects a mass request inconsistent with
    `constants.ion_mass`.
    """
    active_mass = float(constants.ion_mass)
    if ion_mass_kg is None:
        return active_mass
    requested = float(ion_mass_kg)
    if not np.isclose(requested, active_mass, rtol=1e-12, atol=0.0):
        raise ValueError(
            "build_reachability_model mass mismatch: ion_mass_kg="
            f"{requested} but active constants.ion_mass={active_mass}. "
            "Current A-building uses simulation-side global mass; pass None or the "
            "active constants.ion_mass value for honest behavior."
        )
    return requested


def _resolve_conversion_charge(
    *,
    ion_charge_c: float | None,
    poly_is_potential_energy: bool,
) -> float | None:
    if poly_is_potential_energy:
        return None
    if ion_charge_c is None:
        raise ValueError(
            "ion_charge_c is required for frequency conversion when "
            "poly_is_potential_energy is False."
        )
    return float(ion_charge_c)


def _normalize_u_bounds(
    u_bounds: Sequence[Tuple[float | None, float | None]],
    n_controls: int,
) -> List[Tuple[float | None, float | None]]:
    if len(u_bounds) != n_controls:
        raise ValueError(f"u_bounds must have length {n_controls}")
    out: List[Tuple[float | None, float | None]] = []
    for lo, hi in u_bounds:
        lo_v = None if lo is None else float(lo)
        hi_v = None if hi is None else float(hi)
        if (lo_v is not None) and (hi_v is not None) and (lo_v > hi_v):
            raise ValueError("each u_bound must satisfy lower <= upper")
        out.append((lo_v, hi_v))
    return out


def _resolve_u_bounds(
    *,
    u_bounds: Sequence[Tuple[float | None, float | None]] | None,
    dc_bounds: Tuple[float, float] | Sequence[Tuple[float, float]],
    rf_dc_bounds: Tuple[float, float] | Sequence[Tuple[float, float]],
    s_bounds: Tuple[float, float],
    n_dc: int,
    n_rf: int,
    n_controls: int,
) -> List[Tuple[float | None, float | None]]:
    if n_controls != n_dc + n_rf + 1:
        raise ValueError(
            "n_controls mismatch: expected len(dc_electrodes)+len(rf_dc_electrodes)+1"
        )
    if u_bounds is not None:
        return _normalize_u_bounds(u_bounds, n_controls)

    dc_list = _expand_bounds(dc_bounds, n_dc)
    rf_list = _expand_bounds(rf_dc_bounds, n_rf)
    s_lo, s_hi = float(s_bounds[0]), float(s_bounds[1])
    if s_lo < 0.0:
        s_lo = 0.0
    return dc_list + rf_list + [(s_lo, s_hi)]


def _expand_bounds(
    bounds: Tuple[float, float] | Sequence[Tuple[float, float]],
    n: int,
) -> List[Tuple[float, float]]:
    if (
        isinstance(bounds, (list, tuple))
        and len(bounds) == 2
        and not isinstance(bounds[0], (list, tuple))
    ):
        lo, hi = bounds
        return [(float(lo), float(hi)) for _ in range(n)]
    if isinstance(bounds, (list, tuple)) and len(bounds) == n:
        return [(float(b[0]), float(b[1])) for b in bounds]
    raise ValueError("bounds must be (low, high) or a list of (low, high) per electrode")


def _bounds_to_arrays(
    bounds: Sequence[Tuple[float | None, float | None]],
) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.array(
        [
            -np.inf if bnd[0] is None else float(bnd[0])
            for bnd in bounds
        ],
        dtype=float,
    )
    hi = np.array(
        [
            np.inf if bnd[1] is None else float(bnd[1])
            for bnd in bounds
        ],
        dtype=float,
    )
    return lo, hi
