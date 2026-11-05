import os, tempfile
import numpy as np
from vislam.sim import simulate, SimConfig
from vislam.dataset import Dataset
from vislam.pipeline import run_filter, run_slam
from vislam.metrics import traj_metrics, map_metrics
from vislam.calibration import estimate_time_offset


def _ds(**kw):
    return simulate(SimConfig(path="L", seed=1, **kw))


def test_dataset_roundtrip():
    ds = _ds()
    with tempfile.TemporaryDirectory() as d:
        ds.save(d)
        b = Dataset.load(d)
    assert np.allclose(ds.imu_acc, b.imu_acc, atol=1e-6) and len(ds.obs_id) == len(b.obs_id)
    assert np.allclose(ds.obs_T, b.obs_T, atol=1e-6) and np.allclose(ds.gt_p, b.gt_p, atol=1e-6)


def test_markers_reduce_drift_versus_inertial_and_vio():
    ds = _ds()
    imu = traj_metrics(run_filter(ds, None, False), ds)["ate_rmse_m"]
    vio = traj_metrics(run_filter(ds, None, True), ds)["ate_rmse_m"]
    ekf = traj_metrics(run_filter(ds, ds.map_gt, True), ds)["ate_rmse_m"]
    assert imu > 5 * vio and vio > 2 * ekf and ekf < 0.06


def test_slam_builds_accurate_marker_map_and_closes_loop():
    ds = simulate(SimConfig(path="rect", laps=2, seed=2))
    vio = traj_metrics(run_filter(ds, None, True), ds)
    tr, mk, _ = run_slam(ds)
    m, sl = map_metrics(mk, ds.map_gt), traj_metrics(tr, ds)
    assert m["n_markers_mapped"] >= 7 and m["marker_pos_err_mean_m"] < 0.15
    assert sl["loop_closure_residual_m"] < vio["loop_closure_residual_m"]


def test_relocalisation_from_saved_map():
    ds = simulate(SimConfig(path="reloc", seed=5, blackouts=((0, 2.0),)))
    tr = run_filter(ds, ds.map_gt, True, init_from_marker=True)
    assert 1.5 <= tr.stats["reloc_time_s"] < 4 and traj_metrics(tr, ds)["ate_rmse_m"] < 0.2


def test_outliers_are_gated():
    ds = simulate(SimConfig(path="L", seed=3, p_outlier=0.1))
    tr = run_filter(ds, ds.map_gt, True)
    assert tr.stats["gated"] > 0 and traj_metrics(tr, ds)["ate_rmse_m"] < 0.2


def test_time_offset_estimation():
    t = np.arange(0, 30, 0.01)
    w = np.abs(np.sin(1.3 * t) + 0.5 * np.sin(3.1 * t))
    tc = np.arange(0, 30, 0.05)
    d_true = 0.12                                         # camera event at t shows IMU motion at t + d
    wc = np.interp(tc + d_true, t, w)
    d, score = estimate_time_offset(t, w, tc, wc)
    assert abs(d - d_true) < 0.02 and score > 0.9
