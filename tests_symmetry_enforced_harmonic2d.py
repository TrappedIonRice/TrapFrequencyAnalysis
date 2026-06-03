import numpy as np

from equilibrium_playground.symmetry_enforced_harmonic2d import (
    SymmetryAnsatzCounts,
    SymmetryReducedState,
    decode_raw_symmetry_variables,
    expanded_total_energy,
    expand_symmetric_coordinates,
    normalize_harmonic_trap,
    pack_physical_symmetry_variables,
    symmetry_total_energy,
    symmetry_total_gradient,
    symmetry_total_ion_count,
    unpack_physical_symmetry_variables,
)


def _sorted_rows(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def test_normalize_harmonic_trap_modes_agree():
    direct = normalize_harmonic_trap(omega_x=1.25, omega_z=2.125)
    ratio = normalize_harmonic_trap(alpha=1.7, scale=1.25)

    assert direct.input_mode == "direct_omega"
    assert ratio.input_mode == "alpha_scale"
    assert np.isclose(direct.omega_x, ratio.omega_x)
    assert np.isclose(direct.omega_z, ratio.omega_z)
    assert np.isclose(direct.a_x, ratio.a_x)
    assert np.isclose(direct.a_z, ratio.a_z)


def test_expand_symmetric_coordinates_matches_ansatz_formula():
    state = SymmetryReducedState(
        vert_distances=np.array([0.75, 1.5], dtype=float),
        hor_distances=np.array([1.2], dtype=float),
        free_x=np.array([0.8], dtype=float),
        free_z=np.array([0.6], dtype=float),
    )

    expanded = expand_symmetric_coordinates(state)
    expected = np.array(
        [
            (0.0, 0.0),
            (0.0, 0.75),
            (0.0, -0.75),
            (0.0, 1.5),
            (0.0, -1.5),
            (1.2, 0.0),
            (-1.2, 0.0),
            (0.8, 0.6),
            (-0.8, 0.6),
            (0.8, -0.6),
            (-0.8, -0.6),
        ],
        dtype=float,
    )

    assert expanded.shape == (symmetry_total_ion_count(2, 1, 1), 2)
    np.testing.assert_allclose(_sorted_rows(expanded), _sorted_rows(expected))


def test_decode_raw_symmetry_variables_enforces_positive_ordered_axes():
    counts = SymmetryAnsatzCounts(3, 2, 1)
    raw = np.array([-4.0, -1.0, 0.3, -2.0, 0.4, -0.7, 0.2], dtype=float)

    decoded = decode_raw_symmetry_variables(raw, counts)
    state = decoded.state

    assert np.all(state.vert_distances > 0.0)
    assert np.all(state.hor_distances > 0.0)
    assert np.all(np.diff(state.vert_distances) > 0.0)
    assert np.all(np.diff(state.hor_distances) > 0.0)
    assert np.all(state.free_x > 0.0)
    assert np.all(state.free_z > 0.0)


def test_direct_energy_and_gradient_match_expanded_and_finite_difference():
    counts = SymmetryAnsatzCounts(1, 1, 2)
    trap = normalize_harmonic_trap(alpha=1.6, scale=0.95)
    physical = np.array([0.8, 1.15, 0.6, 1.3, 0.55, 0.95], dtype=float)
    state = unpack_physical_symmetry_variables(physical, counts)

    direct_energy = symmetry_total_energy(state, trap)
    expanded_energy = expanded_total_energy(expand_symmetric_coordinates(state), trap)
    assert np.isclose(direct_energy, expanded_energy, rtol=1.0e-12, atol=1.0e-12)

    analytic = symmetry_total_gradient(state, trap)
    numeric = np.zeros_like(physical)
    eps = 1.0e-7
    for idx in range(physical.size):
        step = np.zeros_like(physical)
        step[idx] = eps
        state_plus = unpack_physical_symmetry_variables(physical + step, counts)
        state_minus = unpack_physical_symmetry_variables(physical - step, counts)
        numeric[idx] = (
            symmetry_total_energy(state_plus, trap) - symmetry_total_energy(state_minus, trap)
        ) / (2.0 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(pack_physical_symmetry_variables(state), physical)
