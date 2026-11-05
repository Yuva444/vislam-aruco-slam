import os
import numpy as np
import pytest
from vislam.sim import simulate, SimConfig
from vislam.pipeline import run_method
from vislam.metrics import traj_metrics
from vislam.replay import build_replay, no_marker_spans, plotly_replay, render_video

METHODS = ["IMU only", "VIO", "VIO + anchors (EKF)"]


@pytest.fixture(scope="module")
def run():
    ds = simulate(SimConfig(path="rect", laps=1, seed=4, blackouts=((6.0, 13.0),), name="t"))
    trajs = {m: run_method(ds, m)[0] for m in METHODS}
    return ds, trajs, build_replay(ds, trajs)


def test_no_marker_spans_found_from_detections():
    t = np.arange(0, 10, 0.5)
    seen = [[1] if (x < 2 or x >= 7) else [] for x in t]      # empty for 2.0 <= t < 7.0
    assert no_marker_spans(t, seen, 1.0) == [(2.0, 7.0)]
    seen[8] = []                                              # a one-frame dropout is not a "gap"
    assert no_marker_spans(t, seen, 1.0) == [(2.0, 7.0)]
    assert no_marker_spans(t, [[]] * len(t), 1.0) == [(0.0, 10.0)]   # trailing run is closed at the end
    assert no_marker_spans(t, [[1]] * len(t), 1.0) == []


def test_replay_highlights_the_injected_blackout(run):
    ds, _, rp = run
    assert any(abs(a - 6.0) < 0.2 and abs(b - 13.0) < 0.2 for a, b in rp.spans)
    k = int(np.searchsorted(rp.t, 9.0))
    assert rp.seen[k] == [] and rp.in_gap(k) is not None


def test_error_matches_metrics_and_view_direction_is_unit_and_forward(run):
    ds, trajs, rp = run
    for n, tr in trajs.items():
        e = rp.methods[n]["err"]
        assert np.isclose(np.sqrt(np.nanmean(e ** 2)), traj_metrics(tr, ds)["ate_rmse_m"])
    assert np.allclose(np.linalg.norm(rp.gt_dir, axis=1), 1)
    # simulated camera sees markers inside +-fov/2 of its optical axis: every detected marker must lie in that cone
    half = np.radians(rp.fov_deg / 2) + 1e-6
    for k, ids in enumerate(rp.seen):
        for i in ids:
            v = rp.marker_pos[i] - rp.gt_p[k]
            assert np.arccos(v @ rp.gt_dir[k] / np.linalg.norm(v)) < half


def test_plotly_frames_slider_and_detection_lines(run):
    _, _, rp = run
    fig = plotly_replay(rp, max_frames=60)
    assert 50 <= len(fig.frames) <= 62
    assert len(fig.layout.sliders[0].steps) == len(fig.frames)
    assert [b.label for b in fig.layout.updatemenus[0].buttons] == ["▶ Play", "⏸ Pause"]
    assert len(fig.layout.shapes) == len(rp.spans)                       # red bands
    by_name = {t.name: i for i, t in enumerate(fig.data) if t.name}
    li = by_name["Line to detected marker"]
    pos = fig.frames[0].traces.index(li)
    seen_x = [f.data[pos].x for f in fig.frames]
    stride = len(rp.t) // len(fig.frames) + 1
    for j, f in enumerate(fig.frames):                                    # a line exists exactly when a marker is detected
        k = min(j * stride, len(rp.t) - 1)
        if rp.seen[k]:
            assert any(v is not None for v in seen_x[j])
    assert any(len(s) > 1 for s in seen_x)
    assert "NO MARKER VISIBLE" in " ".join(f.layout.annotations[0].text for f in fig.frames)


def test_replay_without_ground_truth_degrades_gracefully(run):
    ds, trajs, _ = run
    ds2 = type(ds)(**{**ds.__dict__, "gt_p": None, "gt_R": None})
    rp = build_replay(ds2, trajs)
    assert rp.gt_p is None and all(m["err"] is None for m in rp.methods.values())
    fig = plotly_replay(rp, max_frames=20)
    assert len(fig.frames) > 0


def test_video_is_written(run, tmp_path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    _, _, rp = run
    out = render_video(rp, str(tmp_path / "r.mp4"), fps=10, speed=8, size=(640, 360), max_seconds=6)
    assert out.endswith(".mp4") and os.path.getsize(out) > 5_000
