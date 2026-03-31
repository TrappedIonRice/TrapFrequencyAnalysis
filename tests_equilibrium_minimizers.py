from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import constants
from sim.equilibrium_minimizers import _Cryo2DCopyObjective
from sim.equilibrium_viewer import (
    build_equilibrium_figure,
    build_equilibrium_summary_lines,
)
from sim.equilibrium_seeders import (
    axial_to_cartesian,
    build_cryo2dcopy_seed_context,
    generate_axial_hex_sites,
    generate_quartic2d_seed_candidates,
    get_default_quartic2d_plane,
    get_quartic2d_planar_bounds_arrays,
    hex_shell_population,
    make_cryo2dcopy_triangular_seed,
    make_cryo_triangular_seed_exact,
    make_cryo_triangular_seed,
    minimal_hex_shell_radius,
    seed_fits_within_bounds,
)


class HarmonicTrapOnlySim:
    def __init__(self, *, kx=0.55, ky=2.5, kz=1.6):
        self.kx = float(kx)
        self.ky = float(ky)
        self.kz = float(kz)
        self.trapVariables = SimpleNamespace(dc_key="dc")
        self.center_fits = {}

    def _ensure_dc_center_fit(self, polyfit_deg=4):
        return None

    def evaluate_center_poly(self, x, y, z):
        return 0.5 * (self.kx * x**2 + self.ky * y**2 + self.kz * z**2)

    def evaluate_center_poly_1stderivatives(self, x, y, z):
        return (self.kx * x, self.ky * y, self.kz * z)

    def evaluate_center_poly_2ndderivatives(self, x, y, z):
        return np.array(
            [
                [self.kx, 0.0, 0.0],
                [0.0, self.ky, 0.0],
                [0.0, 0.0, self.kz],
            ],
            dtype=float,
        )

    def get_U_using_polyfit_dimensionless(self, ionpos_flat):
        positions = np.asarray(ionpos_flat, dtype=float).reshape(-1, 3)
        return 0.5 * np.sum(
            self.kx * positions[:, 0] ** 2
            + self.ky * positions[:, 1] ** 2
            + self.kz * positions[:, 2] ** 2
        )

    def get_U_Grad_using_polyfit_dimensionless(self, ionpos_flat):
        positions = np.asarray(ionpos_flat, dtype=float).reshape(-1, 3)
        grad = np.zeros_like(positions)
        grad[:, 0] = self.kx * positions[:, 0]
        grad[:, 1] = self.ky * positions[:, 1]
        grad[:, 2] = self.kz * positions[:, 2]
        return grad.reshape(-1)


def test_hex_shell_population_and_radius_match_formula():
    assert hex_shell_population(0) == 1
    assert hex_shell_population(1) == 7
    assert hex_shell_population(2) == 19
    assert minimal_hex_shell_radius(1) == 0
    assert minimal_hex_shell_radius(7) == 1
    assert minimal_hex_shell_radius(8) == 2
    assert minimal_hex_shell_radius(19) == 2


def test_generate_axial_hex_sites_matches_population():
    for radius in (0, 1, 2, 3):
        sites = generate_axial_hex_sites(radius)
        assert sites.shape == (hex_shell_population(radius), 2)


def test_make_cryo_triangular_seed_centers_com_and_presqueezes_hard_axis():
    alpha_eff = 2.0
    spacing = 1.0
    seed = make_cryo_triangular_seed(
        7,
        alpha_eff=alpha_eff,
        spacing=spacing,
        jitter_fraction=0.0,
    )

    np.testing.assert_allclose(seed.mean(axis=0), np.zeros(2), atol=1.0e-14)
    assert np.max(np.abs(seed[:, 1])) < np.max(np.abs(seed[:, 0]))

    axial_sites = generate_axial_hex_sites(1)
    lattice = axial_to_cartesian(axial_sites, spacing=spacing, alpha_eff=alpha_eff)
    trap_radius = np.sqrt(lattice[:, 0] ** 2 + (alpha_eff * lattice[:, 1]) ** 2)
    assert np.max(trap_radius[np.argsort(trap_radius)[:7]]) <= np.max(
        np.sqrt(seed[:, 0] ** 2 + (alpha_eff * seed[:, 1]) ** 2)
    ) + 1.0e-12


def test_make_cryo_triangular_seed_exact_matches_cryo_selection_order():
    alpha_yx = 1.7
    spacing = 1.25
    seed = make_cryo_triangular_seed_exact(
        8,
        alpha_yx=alpha_yx,
        spacing=spacing,
        jitter=0.0,
    )

    axial_sites = generate_axial_hex_sites(minimal_hex_shell_radius(8))
    lattice = axial_to_cartesian(axial_sites, spacing=spacing, alpha_eff=alpha_yx)
    trap_radius = np.sqrt(lattice[:, 0] ** 2 + (alpha_yx * lattice[:, 1]) ** 2)
    kept = lattice[np.argsort(trap_radius, kind="stable")[:8]].copy()
    kept -= kept.mean(axis=0, keepdims=True)

    np.testing.assert_allclose(seed.points_2d.mean(axis=0), np.zeros(2), atol=1.0e-14)
    np.testing.assert_allclose(seed.u0, seed.points_2d[:, 0])
    np.testing.assert_allclose(seed.v0, seed.points_2d[:, 1])
    np.testing.assert_allclose(seed.uv0, np.concatenate([seed.u0, seed.v0]))
    np.testing.assert_allclose(
        seed.points_2d[np.lexsort((seed.points_2d[:, 1], seed.points_2d[:, 0]))],
        kept[np.lexsort((kept[:, 1], kept[:, 0]))],
    )


def test_make_cryo2dcopy_triangular_seed_matches_literal_defaults():
    seed = make_cryo2dcopy_triangular_seed(9, 1.8)
    expected = make_cryo_triangular_seed_exact(
        9,
        alpha_yx=1.8,
        spacing=1.0,
        jitter=0.02,
        rng=np.random.default_rng(11),
    )
    np.testing.assert_allclose(seed.points_2d, expected.points_2d)
    np.testing.assert_allclose(seed.uv0, expected.uv0)


def test_build_cryo2dcopy_seed_context_uses_lab_axis_ratio():
    sim = HarmonicTrapOnlySim(kx=0.5, ky=3.0, kz=2.0)
    context = build_cryo2dcopy_seed_context(sim)
    assert np.isclose(context.alpha_seed, 2.0)
    np.testing.assert_allclose(context.reference_point_dimless, np.zeros(3))


def test_tiny_jitter_preserves_triangular_seed_shape():
    spacing = 1.1
    seed_no_jitter = make_cryo_triangular_seed(
        12,
        alpha_eff=1.8,
        spacing=spacing,
        jitter_fraction=0.0,
    )
    seed_with_jitter = make_cryo_triangular_seed(
        12,
        alpha_eff=1.8,
        spacing=spacing,
        jitter_fraction=0.0016,
        rng=np.random.default_rng(0),
    )

    rms_shift = np.sqrt(np.mean((seed_with_jitter - seed_no_jitter) ** 2))
    assert rms_shift < 0.01 * spacing


def test_cryo2dcopy_objective_gradient_projects_out_com_mode():
    sim = HarmonicTrapOnlySim(kx=1.0, ky=5.0, kz=4.0)
    objective = _Cryo2DCopyObjective(sim, 3)
    xz = np.array([1.0, -0.5, 0.25, 0.2, -0.1, 0.6], dtype=float)
    grad = objective.gradient(xz)

    assert grad.shape == (6,)
    assert np.isclose(np.sum(grad[:3]), 0.0)
    assert np.isclose(np.sum(grad[3:]), 0.0)


def test_quartic2d_planar_bounds_are_tripled():
    lower, upper = get_quartic2d_planar_bounds_arrays()
    np.testing.assert_allclose(
        lower,
        np.array(
            [-3.0 * constants.center_x_bounds, -3.0 * constants.center_z_bounds],
            dtype=float,
        ),
    )
    np.testing.assert_allclose(
        upper,
        np.array(
            [3.0 * constants.center_x_bounds, 3.0 * constants.center_z_bounds],
            dtype=float,
        ),
    )


def test_seed_fits_within_bounds_rejects_without_distorting():
    seed = make_cryo_triangular_seed_exact(
        7,
        alpha_yx=1.6,
        spacing=0.3,
        jitter=0.0,
    ).points_2d
    assert seed_fits_within_bounds(
        seed,
        np.array([-0.5, -0.5]),
        np.array([0.5, 0.5]),
    )
    shifted = seed + np.array([1.0, 0.0], dtype=float)
    assert not seed_fits_within_bounds(
        shifted,
        np.array([-0.5, -0.5]),
        np.array([0.5, 0.5]),
    )


def test_generate_quartic2d_seed_candidates_is_cryo_only_and_fast():
    sim = HarmonicTrapOnlySim()
    seeds = generate_quartic2d_seed_candidates(
        sim,
        5,
        plane=get_default_quartic2d_plane(),
        rng=np.random.default_rng(0),
    )

    assert len(seeds) == 24
    assert {seed["family"] for seed in seeds} == {"cryo_triangular"}
    assert all(np.asarray(seed["positions_2d"]).shape == (5, 2) for seed in seeds)
    assert len({round(float(seed["spacing"]), 12) for seed in seeds}) == 8
    assert all(seed.get("seed_fits_bounds_directly") is True for seed in seeds)
    assert all("rotation_angle_rad" not in seed for seed in seeds)
    assert all("mirrored" not in seed for seed in seeds)


def test_generate_quartic2d_seed_candidates_rejects_out_of_bounds_spacing():
    sim = HarmonicTrapOnlySim()
    seeds = generate_quartic2d_seed_candidates(
        sim,
        5,
        plane=get_default_quartic2d_plane(),
        spacing_values=(100.0,),
        jitters_per_spacing=1,
        jitter_fraction=0.0,
        rng=np.random.default_rng(0),
    )
    assert seeds == []


def test_equilibrium_viewer_handles_missing_and_present_data():
    assert build_equilibrium_figure(None) is None

    final_positions = np.array(
        [
            [-2.0e-6, 0.0, -1.0e-6],
            [0.0, 0.0, 0.0],
            [2.0e-6, 0.0, 1.0e-6],
        ],
        dtype=float,
    )
    seed_positions = final_positions * 0.8

    fig = build_equilibrium_figure(
        final_positions,
        seed_positions=seed_positions,
        plane_normal=np.array([0.0, 1.0, 0.0]),
    )
    assert fig is not None
    plt.close(fig)

    summary_lines = build_equilibrium_summary_lines(
        {
            "minimizer_name": "Quartic2D_101",
            "success": True,
            "energy": 1.25,
            "seed_family": "cryo_triangular",
            "stage": "basinhopping",
            "projected_gradient_max_norm": 2.0e-7,
            "seed_fits_bounds_directly": True,
            "alpha_eff": 1.25,
            "seed_spacing": 1.0,
            "seed_jitter": 0.02,
            "seed_rng_seed": 11,
        }
    )
    assert any("Quartic2D_101" in line for line in summary_lines)
    assert any("success" in line for line in summary_lines)
    assert any("winner stage" in line for line in summary_lines)
    assert any("seed fits bounds directly" in line for line in summary_lines)
    assert any("seed spacing" in line for line in summary_lines)
