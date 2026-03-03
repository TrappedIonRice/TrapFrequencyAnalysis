import os
import numpy as np

from trap_A_cache import get_or_build_A


def test_cache_hit(tmp_path):
    calls = {"n": 0}

    def fake_builder(**kwargs):
        calls["n"] += 1
        return {"A": np.eye(2), "powers": np.array([[0, 0, 0], [1, 0, 0]], dtype=int)}

    cfg = dict(
        trap_name="T",
        dc_electrodes=["DC1"],
        rf_dc_electrodes=["RF1"],
        rf_freq_hz=1.0,
        polyfit_deg=4,
        dc_bounds=[(-1.0, 1.0)],
        rf_dc_bounds=[(-1.0, 1.0)],
        rf2_bounds=(0.0, 1.0),
        num_samples=2,
        seed=0,
    )

    out1 = get_or_build_A(cache_dir=str(tmp_path), builder_fn=fake_builder, force_rebuild=False, **cfg)
    out2 = get_or_build_A(cache_dir=str(tmp_path), builder_fn=fake_builder, force_rebuild=False, **cfg)

    assert calls["n"] == 1
    assert out1["cache_hit"] is False
    assert out2["cache_hit"] is True
    assert os.path.exists(out2["cache_path"])


def test_force_rebuild(tmp_path):
    calls = {"n": 0}

    def fake_builder(**kwargs):
        calls["n"] += 1
        return {"A": np.eye(2), "powers": np.array([[0, 0, 0], [1, 0, 0]], dtype=int)}

    cfg = dict(
        trap_name="T",
        dc_electrodes=["DC1"],
        rf_dc_electrodes=["RF1"],
        rf_freq_hz=1.0,
        polyfit_deg=4,
        dc_bounds=[(-1.0, 1.0)],
        rf_dc_bounds=[(-1.0, 1.0)],
        rf2_bounds=(0.0, 1.0),
        num_samples=2,
        seed=0,
    )

    get_or_build_A(cache_dir=str(tmp_path), builder_fn=fake_builder, force_rebuild=False, **cfg)
    get_or_build_A(cache_dir=str(tmp_path), builder_fn=fake_builder, force_rebuild=True, **cfg)

    assert calls["n"] == 2
