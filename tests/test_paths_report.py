import numpy as np
import pytest
from vislam.paths import PRESETS, ROOM, path_curve, path_length, random_path, self_intersects, validate_path
from vislam.sim import simulate, SimConfig
from vislam.pipeline import run_method
from vislam.replay import build_replay
from vislam.report import gap_stats, method_table, narrative


def test_presets_are_valid():
    for name, wp in PRESETS.items():
        errs, _ = validate_path(wp)
        assert not errs, (name, errs)


@pytest.mark.parametrize("wp,frag", [([(1, 1)], "at least 2"), ([(1, 1), (1, 1), (3, 3)], "identical"),
                                     ([(0.1, 1), (3, 3)], "wall"), ([(1, 1), (1.5, 1)], "long"),
                                     ([(1, 1), (np.nan, 2)], "empty")])
def test_invalid_paths_are_rejected_with_a_message(wp, frag):
    errs, _ = validate_path(wp)
    assert errs and any(frag in e for e in errs)


def test_random_paths_are_valid_reproducible_and_do_not_cross():
    for seed in range(15):
        for closed in (False, True):
            wp = random_path(seed, 3 + seed % 4, closed)
            assert not validate_path(wp)[0]
            assert not self_intersects(wp)
            assert np.array_equal(wp, random_path(seed, 3 + seed % 4, closed))
            if closed:
                assert np.allclose(wp[0], wp[-1])
    assert not np.array_equal(random_path(1), random_path(2))


def test_curve_matches_the_simulated_walk():
    wp = PRESETS["Zig-zag"]
    ds = simulate(SimConfig(path=wp, seed=0))
    c = path_curve(wp, 2000)
    d = np.min(np.linalg.norm(ds.gt_p[:, None, :2] - c[None], axis=2), axis=1)
    assert d.max() < 0.05 and path_length(wp) > 5


def test_report_narrative_and_gap_stats():
    ds = simulate(SimConfig(path="rect", laps=1, seed=3, blackouts=((6.0, 13.0),)))
    trajs = {m: run_method(ds, m)[0] for m in ["IMU only", "VIO", "VIO + anchors (EKF)"]}
    rp = build_replay(ds, trajs)
    table = method_table(ds, trajs)
    gs = gap_stats(rp)
    assert gs["VIO + anchors (EKF)"]["gaps"] and gs["IMU only"]["worst_in_gap"] > gs["VIO"]["worst_in_gap"]
    text = " ".join(narrative(ds, rp, table))
    assert "Best estimate" in text and "marker-free stretch" in text
    ds2 = simulate(SimConfig(path="L", seed=1))                # no gap -> says so instead of inventing one
    tr2 = {m: run_method(ds2, m)[0] for m in ["IMU only", "VIO + anchors (EKF)"]}
    assert "cannot be seen" in " ".join(narrative(ds2, build_replay(ds2, tr2), method_table(ds2, tr2)))
