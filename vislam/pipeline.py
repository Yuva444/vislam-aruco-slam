"""Replay pipeline: EKF localisation (IMU / VIO / VIO+markers) and graph-SLAM mapping."""
import time
from dataclasses import dataclass, field
import numpy as np
from .ekf import ESEKF, EKFConfig
from .geometry import gravity_alignment, make_T, inv_T
from .graph import optimise_graph


@dataclass
class Trajectory:
    p: np.ndarray                    # (N,3) per camera frame
    R: np.ndarray                    # (N,3,3)
    ok: np.ndarray                   # bool, frame has a valid estimate
    pos_std: np.ndarray = None
    stats: dict = field(default_factory=dict)
    name: str = ""


def initial_pose(ds):
    if "start_pose" in ds.meta:
        return np.array(ds.meta["start_pose"])
    n = max(10, int(0.5 * ds.meta.get("imu_rate", 100)))
    return make_T(gravity_alignment(ds.imu_acc[:n].mean(0)), np.zeros(3))


def run_filter(ds, marker_map=None, use_vo=True, init_pose=None, init_from_marker=False,
               cfg=EKFConfig(), name=""):
    """Replay a dataset through the error-state EKF.

    marker_map=None  -> no marker corrections (inertial / VIO baseline)
    init_from_marker -> pose unknown at start: relocalise from the first detected map marker
    """
    t0 = time.perf_counter()
    T_BC = ds.T_BC
    obs, vo = ds.obs_by_frame(), ds.vo_by_frame()
    vo_dir = ds.meta.get("vo_kind", "velocity") == "direction"
    nF = len(ds.cam_t)
    P, Rm = np.full((nF, 3), np.nan), np.tile(np.eye(3), (nF, 1, 1))
    ok, sd = np.zeros(nF, bool), np.full(nF, np.nan)
    ekf = ESEKF(init_pose if init_pose is not None else initial_pose(ds), cfg)
    initialised = not init_from_marker
    t_loc, n_det_used = None, 0

    def frame_update(fi):
        nonlocal initialised, t_loc, n_det_used
        if marker_map is not None:
            ms = [(i, T) for i, T in obs.get(fi, []) if i in marker_map]
            if not initialised and ms:
                i, T = min(ms, key=lambda m: np.linalg.norm(m[1][:3, 3]))
                ekf.reset_pose(marker_map[i] @ inv_T(T) @ inv_T(T_BC))
                initialised, t_loc = True, ds.cam_t[fi]
            elif initialised:
                for i, T in ms:
                    n_det_used += ekf.marker_pose(marker_map[i], T, T_BC)
        if initialised and use_vo and fi in vo:
            (ekf.vo_direction if vo_dir else ekf.vo_velocity)(vo[fi])
        if initialised:
            P[fi], Rm[fi], ok[fi] = ekf.p, ekf.R, True
            sd[fi] = np.sqrt(np.trace(ekf.P[:3, :3]) / 3)

    fi = 0
    for i in range(len(ds.imu_t) - 1):
        a, g = ds.imu_acc[i], ds.imu_gyr[i]
        ta, tb = ds.imu_t[i], ds.imu_t[i + 1]
        while fi < nF and ds.cam_t[fi] <= tb + 1e-9:
            tf = ds.cam_t[fi]
            if tf > ta and initialised:
                ekf.predict(a, g, tf - ta)
            ta = max(ta, tf)
            frame_update(fi)
            fi += 1
        if tb > ta and initialised:
            ekf.predict(a, g, tb - ta)
    stats = {"runtime_s": time.perf_counter() - t0, "marker_updates": int(ekf.n_acc), "gated": ekf.n_rej,
             "reloc_time_s": None if t_loc is None else float(t_loc - ds.cam_t[0]) if init_from_marker else 0.0}
    return Trajectory(P, Rm, ok, sd, stats, name)


def run_slam(ds, kf_every=10, init_pose=None, cfg=EKFConfig(), verbose=False):
    """VIO front-end + factor-graph back-end.  Marker poses are *unknown* landmarks.

    Returns (Trajectory, marker_map {id: T_WM}, info)."""
    t0 = time.perf_counter()
    vio = run_filter(ds, None, True, init_pose, cfg=cfg, name="vio")
    T_BC = ds.T_BC
    obs = ds.obs_by_frame()
    nF = len(ds.cam_t)
    kfs = sorted(set(range(0, nF, kf_every)) | {nF - 1})
    kf_arr = np.array(kfs)
    T_vio = [make_T(vio.R[f], vio.p[f]) for f in range(nF)]
    T_odo = np.array([T_vio[f] for f in kfs])
    T0 = np.array(init_pose if init_pose is not None else initial_pose(ds))
    meas = []                                   # marker obs attached to the nearest keyframe
    for f, lst in obs.items():
        k = int(np.abs(kf_arr - f).argmin())
        T_rel = inv_T(T_vio[kfs[k]]) @ T_vio[f]
        meas += [(k, i, T, T_rel) for i, T in lst]
    T_kf, mk, info = optimise_graph(T_odo, meas, T_BC, T0, verbose=verbose)
    # propagate the keyframe correction to every frame with the local VIO motion
    P, Rm = np.zeros((nF, 3)), np.zeros((nF, 3, 3))
    for f in range(nF):
        j = np.searchsorted(kf_arr, f, side="right") - 1
        Tj = T_kf[j] @ inv_T(T_odo[j]) @ T_vio[f]
        P[f], Rm[f] = Tj[:3, 3], Tj[:3, :3]
    info.update(runtime_s=time.perf_counter() - t0, n_keyframes=len(kfs))
    traj = Trajectory(P, Rm, np.ones(nF, bool), None, {"runtime_s": info["runtime_s"]}, "slam")
    return traj, mk, info


METHODS = ["IMU only", "VIO", "VIO + anchors (EKF)", "VIO + landmarks (graph SLAM)", "Relocalisation (map + EKF)"]


def run_method(ds, method, marker_map=None):
    """Run one named method on a dataset.  Returns (Trajectory, estimated marker map or None, graph info or None).

    marker_map is the (surveyed / saved) map used by the anchor and relocalisation methods; default: ds.map_gt."""
    amap = marker_map if marker_map is not None else ds.map_gt
    if method == "IMU only":
        return run_filter(ds, None, False), None, None
    if method == "VIO":
        return run_filter(ds, None, True), None, None
    if method == "VIO + anchors (EKF)":
        return run_filter(ds, amap, True), None, None
    if method == "Relocalisation (map + EKF)":
        return run_filter(ds, amap, True, init_from_marker=True), None, None
    if method == "VIO + landmarks (graph SLAM)":
        return run_slam(ds)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
