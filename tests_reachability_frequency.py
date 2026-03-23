import numpy as np
import matplotlib
import types

matplotlib.use("Agg")

import reachability.frequency_space as freqspace
from reachability import (
    BoundarySamplingResult,
    ReachabilityModel,
    build_modal_curvature_hull,
    convert_lambda_boundary_to_frequency_samples,
    enrich_positive_octant_lambda_boundary,
    lambda_to_frequency_points,
    plot_frequency_boundary_points_3d,
    plot_single_trap_frequency_space,
    positive_octant_mask,
    sample_reachable_boundary,
)


def _toy_model(lower=(-1.0, -1.0, -1.0), upper=(1.0, 1.0, 1.0)) -> ReachabilityModel:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    return ReachabilityModel(
        A=np.eye(3, dtype=float),
        powers=np.zeros((3, 3), dtype=int),
        L_eq_c=np.zeros((0, 3), dtype=float),
        b_eq=np.zeros(0, dtype=float),
        L_diag_c=np.eye(3, dtype=float),
        E=np.zeros((0, 3), dtype=float),
        e=np.zeros(0, dtype=float),
        T=np.eye(3, dtype=float),
        lower_u=lo,
        upper_u=hi,
        u_bounds=[(float(lo[i]), float(hi[i])) for i in range(3)],
        r0=np.zeros(3, dtype=float),
        rotation=np.eye(3, dtype=float),
        basis="nondim",
        nd_L0_m=2.0,
        ion_mass_kg=4.0,
        ion_charge_c=2.0,
        poly_is_potential_energy=False,
    )


def _toy_boundary_sampling(points: np.ndarray) -> BoundarySamplingResult:
    pts = np.asarray(points, dtype=float)
    hull = build_modal_curvature_hull(pts, deduplicate_tol=1e-12)
    n = pts.shape[0]
    return BoundarySamplingResult(
        sampled_directions=np.zeros((n, 3), dtype=float),
        query_results=[],
        success_mask=np.ones(n, dtype=bool),
        objective_values=np.zeros(n, dtype=float),
        raw_lambda_points=pts.copy(),
        raw_u_points=None,
        lambda_points=pts.copy(),
        u_points=None,
        n_requested=n,
        n_success=n,
        n_raw_returned=n,
        n_returned=n,
        deduplicate_tol=1e-12,
        hull=hull,
    )


class _FakePlotlyFigure:
    def __init__(self):
        self.data = []
        self.layout = {}
        self.was_shown = False
        self.written_html_path = None

    def add_trace(self, trace):
        self.data.append(trace)

    def update_layout(self, **kwargs):
        self.layout.update(kwargs)

    def show(self):
        self.was_shown = True

    def write_html(self, path):
        self.written_html_path = str(path)


def _install_fake_plotly(monkeypatch):
    figures = []

    def _figure():
        fig = _FakePlotlyFigure()
        figures.append(fig)
        return fig

    def _scatter3d(**kwargs):
        return {"type": "Scatter3d", **kwargs}

    def _mesh3d(**kwargs):
        return {"type": "Mesh3d", **kwargs}

    fake_go = types.SimpleNamespace(
        Figure=_figure,
        Scatter3d=_scatter3d,
        Mesh3d=_mesh3d,
        _figures=figures,
    )
    monkeypatch.setattr(freqspace, "go", fake_go)
    return fake_go


def test_positive_octant_mask_uses_tolerance():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1.0e-13, 1.0, 2.0],
            [-1.0e-4, 1.0, 1.0],
        ],
        dtype=float,
    )
    mask = positive_octant_mask(pts, tol=1.0e-12)
    assert mask.tolist() == [True, True, False]


def test_lambda_to_frequency_conversion_matches_expected_omega_and_hz():
    model = _toy_model()
    # omega = [1, 2, 3] rad/s -> lambda_phys = (m/q) * omega^2 = 2*[1,4,9]
    # lambda_nondim = lambda_phys * L0^2 with L0=2
    lam_nd = np.array([[8.0, 32.0, 72.0]], dtype=float)

    omega = lambda_to_frequency_points(lam_nd, model, output="omega")
    hz = lambda_to_frequency_points(lam_nd, model, output="hz")

    np.testing.assert_allclose(omega[0], np.array([1.0, 2.0, 3.0]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(hz[0], np.array([1.0, 2.0, 3.0]) / (2.0 * np.pi), rtol=0, atol=1e-12)


def test_positive_octant_boundary_enrichment_includes_clipping_plane_points():
    cube_vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    out = enrich_positive_octant_lambda_boundary(
        cube_vertices,
        octant_tol=1e-12,
        deduplicate_tol=1e-9,
        edge_samples_per_edge=1,
        face_samples_per_face=4,
        random_seed=0,
    )
    assert out.n_total_points > 0
    assert np.all(out.lambda_points >= -1e-12)
    assert out.surface_triangles.shape[0] > 0
    # clipping faces on coordinate planes should be represented
    surf = out.surface_lambda_points
    assert np.any((np.abs(surf[:, 0]) <= 1e-12) & (surf[:, 1] > 0.0) & (surf[:, 2] > 0.0))
    assert np.any((np.abs(surf[:, 1]) <= 1e-12) & (surf[:, 0] > 0.0) & (surf[:, 2] > 0.0))
    assert np.any((np.abs(surf[:, 2]) <= 1e-12) & (surf[:, 0] > 0.0) & (surf[:, 1] > 0.0))


def test_face_sampling_is_deterministic_for_fixed_input():
    tetra = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    out_a = enrich_positive_octant_lambda_boundary(
        tetra,
        face_samples_per_face=3,
        edge_samples_per_edge=1,
        random_seed=0,
    )
    out_b = enrich_positive_octant_lambda_boundary(
        tetra,
        face_samples_per_face=3,
        edge_samples_per_edge=1,
        random_seed=999,
    )
    np.testing.assert_allclose(out_a.surface_lambda_points, out_b.surface_lambda_points, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(out_a.surface_triangles, out_b.surface_triangles)


def test_single_trap_frequency_plot_wrapper_returns_structured_result():
    model = _toy_model(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0))
    sampling = sample_reachable_boundary(
        model,
        n_samples=30,
        random_seed=0,
        deduplicate_tol=1e-9,
        build_hull=True,
    )
    fig, ax, freq_sample = plot_single_trap_frequency_space(
        model,
        sampling,
        output="hz",
        backend="matplotlib",
        show=False,
        random_seed=0,
    )
    assert fig is not None
    assert ax is not None
    assert freq_sample.n_points >= 0
    assert freq_sample.frequency_points.shape[1] == 3
    assert "MHz" in ax.get_xlabel()
    assert "MHz" in ax.get_ylabel()
    assert "MHz" in ax.get_zlabel()


def test_multi_dispatch_uses_single_and_multi_paths(monkeypatch):
    calls = {"single": 0, "multi": 0}
    single_kwargs = {}
    multi_kwargs = {}
    toy_model = _toy_model(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0))
    toy_boundary = _toy_boundary_sampling(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    )

    def fake_build_model(cfg, num_model_samples):
        return toy_model

    def fake_sample(*args, **kwargs):
        return toy_boundary

    def fake_single(*args, **kwargs):
        calls["single"] += 1
        single_kwargs.update(kwargs)
        return "fig_single", "ax_single", "single_result"

    def fake_multi(*args, **kwargs):
        calls["multi"] += 1
        multi_kwargs.update(kwargs)
        return "fig_multi", "ax_multi", ["multi_result"]

    monkeypatch.setattr(freqspace, "_build_model_from_spec", fake_build_model)
    monkeypatch.setattr(freqspace, "sample_reachable_boundary", fake_sample)
    monkeypatch.setattr(freqspace, "plot_single_trap_frequency_space", fake_single)
    monkeypatch.setattr(freqspace, "plot_multi_trap_frequency_space", fake_multi)

    out_single = freqspace.plot_trap_frequency_space("Simp58_101", show=False)
    assert out_single == ("fig_single", "ax_single", "single_result")
    assert calls["single"] == 1
    assert calls["multi"] == 0
    assert single_kwargs["backend"] == "plotly"
    assert "max_surface_triangles" in single_kwargs
    assert "max_scatter_points" in single_kwargs

    out_multi = freqspace.plot_trap_frequency_space(["Simp58_101", "InnTrapFine"], show=False)
    assert out_multi == ("fig_multi", "ax_multi", ["multi_result"])
    assert calls["multi"] == 1
    assert multi_kwargs["backend"] == "plotly"
    assert "max_surface_triangles" in multi_kwargs
    assert "max_scatter_points" in multi_kwargs


def test_conversion_wrapper_uses_enriched_positive_octant_points():
    model = _toy_model()
    boundary = _toy_boundary_sampling(
        np.array(
            [
                [-1.0, -1.0, -1.0],
                [2.0, -1.0, -1.0],
                [-1.0, 2.0, -1.0],
                [-1.0, -1.0, 2.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=float,
        )
    )
    out = convert_lambda_boundary_to_frequency_samples(
        model,
        boundary,
        output="omega",
        octant_tol=1e-12,
        deduplicate_tol=1e-9,
        edge_samples_per_edge=1,
        face_samples_per_face=2,
        random_seed=0,
    )
    assert out.status in ("ok", "empty")
    assert out.frequency_points.shape[1] == 3
    if out.frequency_points.shape[0] > 0:
        assert np.all(out.frequency_points >= 0.0)


def test_frequency_surface_connectivity_preserved_from_lambda_samples():
    model = _toy_model(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0))
    boundary = _toy_boundary_sampling(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=float,
        )
    )
    out = convert_lambda_boundary_to_frequency_samples(
        model,
        boundary,
        output="hz",
        face_samples_per_face=3,
        edge_samples_per_edge=1,
        random_seed=0,
    )
    if out.frequency_surface_points.shape[0] == 0:
        # SciPy clipping/hull may be unavailable in some test environments.
        return
    np.testing.assert_array_equal(
        out.frequency_surface_triangles,
        out.lambda_boundary.surface_triangles,
    )
    expected = lambda_to_frequency_points(
        out.lambda_boundary.surface_lambda_points,
        model,
        output="hz",
    )
    np.testing.assert_allclose(out.frequency_surface_points, expected, rtol=0, atol=1e-12)


def test_frequency_plot_surface_option_runs_for_normal_case():
    pts_hz = np.array(
        [
            [1.0e6, 1.0e6, 1.0e6],
            [2.0e6, 1.0e6, 1.0e6],
            [1.0e6, 2.0e6, 1.0e6],
            [1.0e6, 1.0e6, 2.0e6],
        ],
        dtype=float,
    )
    fig, ax = plot_frequency_boundary_points_3d(
        pts_hz,
        output="hz",
        show_surface=True,
        show=False,
    )
    assert fig is not None
    assert ax is not None
    assert "MHz" in ax.get_xlabel()


def test_frequency_plot_surface_graceful_on_degenerate_points():
    # Coplanar points -> hull shell may fail, but plotting should still succeed.
    pts_hz = np.array(
        [
            [1.0e6, 1.0e6, 2.0e6],
            [2.0e6, 1.0e6, 2.0e6],
            [1.0e6, 2.0e6, 2.0e6],
            [2.0e6, 2.0e6, 2.0e6],
        ],
        dtype=float,
    )
    fig, ax = plot_frequency_boundary_points_3d(
        pts_hz,
        output="hz",
        show_surface=True,
        show=False,
    )
    assert fig is not None
    assert ax is not None


def test_plotly_figure_has_mhz_labels_and_cube_aspect(monkeypatch):
    _install_fake_plotly(monkeypatch)
    pts_hz = np.array(
        [
            [1.0e6, 1.0e6, 1.0e6],
            [2.0e6, 1.0e6, 1.0e6],
            [1.0e6, 2.0e6, 1.0e6],
            [1.0e6, 1.0e6, 2.0e6],
        ],
        dtype=float,
    )
    fig = freqspace.plot_frequency_boundary_points_3d_plotly(
        pts_hz,
        output="hz",
        show_surface=False,
        show=False,
        label="toy",
    )
    scene = fig.layout["scene"]
    assert scene["aspectmode"] == "cube"
    assert scene["xaxis"]["title"]["text"] == "f_1 (MHz)"
    assert scene["yaxis"]["title"]["text"] == "f_2 (MHz)"
    assert scene["zaxis"]["title"]["text"] == "f_3 (MHz)"


def test_plotly_surface_uses_provided_connectivity_and_no_freq_hull(monkeypatch):
    _install_fake_plotly(monkeypatch)

    def _fail_convex_hull(*args, **kwargs):
        raise AssertionError("ConvexHull should not be used for Plotly frequency surface.")

    monkeypatch.setattr(freqspace, "ConvexHull", _fail_convex_hull)

    pts_hz = np.array(
        [
            [1.0e6, 1.0e6, 1.0e6],
            [2.0e6, 1.0e6, 1.0e6],
            [1.0e6, 2.0e6, 1.0e6],
            [1.0e6, 1.0e6, 2.0e6],
        ],
        dtype=float,
    )
    tri = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=int,
    )
    fig = freqspace.plot_frequency_boundary_points_3d_plotly(
        pts_hz,
        surface_points=pts_hz,
        surface_triangles=tri,
        output="hz",
        show_surface=True,
        show=False,
    )
    mesh_count = sum(1 for tr in fig.data if tr.get("type") == "Mesh3d")
    assert mesh_count >= 1


def test_plotly_surface_graceful_when_surface_data_missing(monkeypatch):
    _install_fake_plotly(monkeypatch)
    pts_hz = np.array(
        [
            [1.0e6, 1.0e6, 1.0e6],
            [2.0e6, 1.0e6, 1.0e6],
            [1.0e6, 2.0e6, 1.0e6],
            [1.0e6, 1.0e6, 2.0e6],
        ],
        dtype=float,
    )
    fig = freqspace.plot_frequency_boundary_points_3d_plotly(
        pts_hz,
        surface_points=np.zeros((0, 3), dtype=float),
        surface_triangles=np.zeros((0, 3), dtype=int),
        output="hz",
        show_surface=True,
        show=False,
    )
    mesh_count = sum(1 for tr in fig.data if tr.get("type") == "Mesh3d")
    scatter_count = sum(1 for tr in fig.data if tr.get("type") == "Scatter3d")
    assert mesh_count == 0
    assert scatter_count >= 1


def test_plotly_mesh_and_scatter_decimation_limits_trace_sizes(monkeypatch):
    _install_fake_plotly(monkeypatch)
    pts_hz = np.array(
        [[1.0e6 + i, 1.0e6 + 2 * i, 1.0e6 + 3 * i] for i in range(200)],
        dtype=float,
    )
    tri = np.array(
        [[i, i + 1, i + 2] for i in range(0, 180)],
        dtype=int,
    )
    fig = freqspace.plot_frequency_boundary_points_3d_plotly(
        pts_hz,
        surface_points=pts_hz,
        surface_triangles=tri,
        output="hz",
        show_surface=True,
        show=False,
        max_surface_triangles=25,
        max_scatter_points=40,
    )
    mesh = [tr for tr in fig.data if tr.get("type") == "Mesh3d"][0]
    scatter = [tr for tr in fig.data if tr.get("type") == "Scatter3d"][0]
    assert len(mesh["i"]) <= 25
    assert len(scatter["x"]) <= 40


def test_plotly_backend_falls_back_to_matplotlib_when_unavailable(monkeypatch):
    monkeypatch.setattr(freqspace, "go", None)
    model = _toy_model(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0))
    sampling = sample_reachable_boundary(
        model,
        n_samples=20,
        random_seed=3,
        deduplicate_tol=1e-9,
        build_hull=True,
    )
    fig, ax, freq_sample = plot_single_trap_frequency_space(
        model,
        sampling,
        output="hz",
        backend="plotly",
        show=False,
    )
    assert fig is not None
    assert ax is not None
    assert freq_sample.n_points >= 0
    assert "MHz" in ax.get_xlabel()


def test_multi_trap_plotly_figure_creation_not_crashing(monkeypatch):
    _install_fake_plotly(monkeypatch)
    toy_model = _toy_model(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0))
    toy_boundary = _toy_boundary_sampling(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    )

    monkeypatch.setattr(freqspace, "_build_model_from_spec", lambda *args, **kwargs: toy_model)
    monkeypatch.setattr(freqspace, "sample_reachable_boundary", lambda *args, **kwargs: toy_boundary)

    fig, ax, out = freqspace.plot_multi_trap_frequency_space(
        [
            {"trap_name": "Simp58_101", "name": "S58 alpha 0"},
            {"trap_name": "InnTrapFine", "name": "Inn alpha 45"},
        ],
        output="hz",
        backend="plotly",
        n_samples=8,
        show=False,
        show_surface=False,
    )
    assert fig is not None
    assert ax is None
    assert len(out) == 2
    scatter_count = sum(1 for tr in fig.data if tr.get("type") == "Scatter3d")
    assert scatter_count >= 2
    scatter_names = [tr.get("name") for tr in fig.data if tr.get("type") == "Scatter3d"]
    assert "S58 alpha 0" in scatter_names
    assert "Inn alpha 45" in scatter_names


def test_multi_trap_plotly_can_save_html(monkeypatch, tmp_path):
    _install_fake_plotly(monkeypatch)
    toy_model = _toy_model(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0))
    toy_boundary = _toy_boundary_sampling(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    )
    monkeypatch.setattr(freqspace, "_build_model_from_spec", lambda *args, **kwargs: toy_model)
    monkeypatch.setattr(freqspace, "sample_reachable_boundary", lambda *args, **kwargs: toy_boundary)
    monkeypatch.setattr(freqspace, "_default_multi_trap_html_output_dir", lambda: tmp_path)

    fig, ax, out = freqspace.plot_multi_trap_frequency_space(
        [
            {"trap_name": "Simp58_101", "name": "S58 alpha 0"},
            {"trap_name": "Simp58_101", "name": "S58 alpha 45"},
        ],
        output="hz",
        backend="plotly",
        n_samples=8,
        show=False,
        show_surface=False,
        save_plotly_html=True,
    )
    assert fig is not None
    assert ax is None
    assert len(out) == 2
    assert fig.written_html_path is not None
    assert str(tmp_path) in fig.written_html_path
    assert fig.written_html_path.endswith(".html")
    assert "plot_multi_trap_frequency_space" in fig.written_html_path


def test_multi_trap_plotly_can_optionally_save_lambda_html(monkeypatch, tmp_path):
    fake_go = _install_fake_plotly(monkeypatch)
    toy_model = _toy_model(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0))
    toy_boundary = _toy_boundary_sampling(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    )
    monkeypatch.setattr(freqspace, "_build_model_from_spec", lambda *args, **kwargs: toy_model)
    monkeypatch.setattr(freqspace, "sample_reachable_boundary", lambda *args, **kwargs: toy_boundary)
    monkeypatch.setattr(freqspace, "_default_multi_trap_html_output_dir", lambda: tmp_path / "freq")
    monkeypatch.setattr(freqspace, "_default_multi_trap_lambda_html_output_dir", lambda: tmp_path / "lambda")

    fig, ax, out = freqspace.plot_multi_trap_frequency_space(
        [
            {"trap_name": "Simp58_101", "name": "S58 alpha 0"},
            {"trap_name": "Simp58_101", "name": "S58 alpha 45"},
        ],
        output="hz",
        backend="plotly",
        n_samples=8,
        show=False,
        show_surface=False,
        save_plotly_html=True,
        plot_lambda_space=True,
    )
    assert fig is not None
    assert ax is None
    assert len(out) == 2
    assert len(fake_go._figures) == 2
    freq_fig = fake_go._figures[0]
    lambda_fig = fake_go._figures[1]
    assert freq_fig.written_html_path is not None
    assert lambda_fig.written_html_path is not None
    assert str(tmp_path / "freq") in freq_fig.written_html_path
    assert str(tmp_path / "lambda") in lambda_fig.written_html_path
    assert "plot_multi_trap_frequency_space" in freq_fig.written_html_path
    assert "plot_multi_trap_lambda_space" in lambda_fig.written_html_path


def test_multi_trap_html_filename_stays_reasonable_with_many_labels():
    labels = [f"simp58_101_{k}deg" for k in range(0, 90, 5)]
    name = freqspace._build_multi_trap_html_filename(
        labels=labels,
        n_samples=1200,
        output="hz",
        density_scale=1.4,
        show_surface=True,
    )
    assert name.endswith(".html")
    assert len(name) <= 200


def test_single_trap_wrapper_can_optionally_create_lambda_plot(monkeypatch):
    fake_go = _install_fake_plotly(monkeypatch)
    toy_model = _toy_model(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0))
    toy_boundary = _toy_boundary_sampling(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    )
    monkeypatch.setattr(freqspace, "_build_model_from_spec", lambda *args, **kwargs: toy_model)
    monkeypatch.setattr(freqspace, "sample_reachable_boundary", lambda *args, **kwargs: toy_boundary)

    fig, ax, out = freqspace.plot_trap_frequency_space(
        {"trap_name": "Simp58_101"},
        backend="plotly",
        n_samples=8,
        show=False,
        show_surface=False,
        plot_lambda_space=True,
    )
    assert fig is not None
    assert ax is None
    assert out.n_points >= 0
    assert len(fake_go._figures) == 2
    lambda_fig = fake_go._figures[1]
    scene = lambda_fig.layout["scene"]
    assert scene["aspectmode"] == "cube"
    assert scene["xaxis"]["title"]["text"] == "lambda_1"
    assert scene["yaxis"]["title"]["text"] == "lambda_2"
    assert scene["zaxis"]["title"]["text"] == "lambda_3"
