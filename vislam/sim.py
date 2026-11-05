"""Physics-based simulator: a walking person with a handheld camera+IMU in a room with ArUco markers.

Generates *exactly the same dataset format* as real recordings, plus ground truth, so every
algorithm is developed and evaluated on repeatable data before field experiments.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as Rot
from .dataset import Dataset
from .geometry import GRAVITY, T_BC_DEFAULT, make_T, inv_T, Exp

ROOM = (6.0, 5.0, 2.8)
# marker slots on the walls: id -> (x, y, normal_x, normal_y, height)
SLOTS = {0: (1.5, 0, 0, 1, 1.3), 1: (4.5, 0, 0, 1, 1.7), 2: (6, 1.5, -1, 0, 1.4), 3: (6, 3.5, -1, 0, 1.2),
         4: (4.5, 5, 0, -1, 1.6), 5: (1.5, 5, 0, -1, 1.3), 6: (0, 3.5, 1, 0, 1.5), 7: (0, 1.5, 1, 0, 1.2)}
PATHS = {
    "L": [(1.5, 1.5), (4.5, 1.5), (4.5, 3.8)],
    "rect": [(1.5, 1.5), (4.5, 1.5), (4.5, 3.5), (1.5, 3.5), (1.5, 1.5)],
    "reloc": [(2.3, 1.2), (4.2, 2.2), (3.2, 3.8), (1.4, 2.6)],
}


@dataclass
class SimConfig:
    path: str = "rect"
    laps: int = 1
    seed: int = 0
    speed: float = 0.8
    ramp: float = 1.5
    sway_deg: float = 40.0           # yaw scanning amplitude of the handheld camera
    marker_ids: tuple = tuple(range(8))
    marker_size: float = 0.18
    imu_rate: float = 200.0
    cam_rate: float = 20.0
    imu_noise_scale: float = 1.0
    acc_density: float = 0.01        # m/s^2/sqrt(Hz)
    gyr_density: float = 0.001       # rad/s/sqrt(Hz)
    acc_rw: float = 0.001
    gyr_rw: float = 1e-4
    acc_bias0: float = 0.05
    gyr_bias0: float = 0.005
    max_range: float = 3.5
    fov_deg: float = 70.0
    max_incidence_deg: float = 70.0
    det_noise_scale: float = 1.0
    p_miss: float = 0.1
    p_outlier: float = 0.0
    blackouts: tuple = ()            # ((t0, t1), ...) windows with no marker visible
    vo_sigma: float = 0.05           # m/s
    vo_scale_sigma: float = 0.05     # initial scale error of the VO source
    name: str = "sim"


def make_markers(ids, size=0.18):
    out = {}
    for i in ids:
        x, y, nx, ny, z = SLOTS[i]
        n = np.array([nx, ny, 0.0])
        up = np.array([0, 0, 1.0])
        xm = np.cross(up, n)
        R = np.column_stack([xm, up, n])
        out[i] = make_T(R, [x, y, z])
    return out


def _perturb(T, d, cfg, rng):
    sp = (0.004 + 0.008 * d) * cfg.det_noise_scale
    sr = np.radians(1.0 + 0.7 * d) * cfg.det_noise_scale
    n = rng.normal(0, sp, 3)
    n[2] *= 3.0                       # depth is the weakest direction for a monocular PnP
    R = Exp(rng.normal(0, sr, 3)) @ T[:3, :3]
    out = make_T(R, T[:3, 3] + n)
    if rng.random() < cfg.p_outlier:  # gross false detection / ambiguity flip
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        out = make_T(Exp(ax * np.radians(rng.uniform(25, 60))) @ out[:3, :3],
                     out[:3, 3] + rng.normal(0, 0.5, 3))
    return out


def simulate(cfg: SimConfig = SimConfig()) -> Dataset:
    rng = np.random.default_rng(cfg.seed)
    wp = np.array(PATHS[cfg.path] if isinstance(cfg.path, str) else cfg.path, float)
    if cfg.laps > 1:
        wp = np.vstack([wp] + [wp[1:]] * (cfg.laps - 1))
    u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    spl = CubicSpline(u, wp, bc_type="natural")
    L = u[-1]
    T = L / cfg.speed + cfg.ramp
    dt = 1.0 / cfg.imu_rate
    t = np.arange(0, T, dt)
    vprof = cfg.speed * np.clip(np.minimum(t / cfg.ramp, (T - t) / cfg.ramp), 0, 1)
    s = np.clip(np.cumsum(vprof) * dt, 0, L)
    sdd = np.gradient(vprof, dt)
    P, dP, ddP = spl(s), spl(s, 1), spl(s, 2)
    p = np.column_stack([P, 1.5 + 0.015 * np.sin(2 * np.pi * 1.8 * t)])
    vel = np.column_stack([dP * vprof[:, None], 0.015 * 2 * np.pi * 1.8 * np.cos(2 * np.pi * 1.8 * t)])
    acc = np.column_stack([ddP * (vprof ** 2)[:, None] + dP * sdd[:, None],
                           -0.015 * (2 * np.pi * 1.8) ** 2 * np.sin(2 * np.pi * 1.8 * t)])
    heading = np.unwrap(np.arctan2(dP[:, 1], dP[:, 0]))
    ph = rng.uniform(0, 2 * np.pi)
    yaw = heading + np.radians(cfg.sway_deg) * np.sin(2 * np.pi * t / 6.5 + ph)
    pitch = np.radians(4) * np.sin(2 * np.pi * t / 4.1)
    roll = np.radians(3) * np.sin(2 * np.pi * t / 3.3 + 1)
    rots = Rot.from_euler("ZYX", np.column_stack([yaw, pitch, roll]))
    Rm = rots.as_matrix()
    w_b = (rots[:-1].inv() * rots[1:]).as_rotvec() / dt
    w_b = np.vstack([w_b, w_b[-1]])
    f_b = np.einsum("nji,nj->ni", Rm, acc - GRAVITY)

    k = cfg.imu_noise_scale
    n = len(t)
    ba = cfg.acc_bias0 * rng.normal(size=3) + np.cumsum(rng.normal(0, cfg.acc_rw * k * np.sqrt(dt), (n, 3)), 0)
    bg = cfg.gyr_bias0 * rng.normal(size=3) + np.cumsum(rng.normal(0, cfg.gyr_rw * k * np.sqrt(dt), (n, 3)), 0)
    imu_acc = f_b + ba + rng.normal(0, cfg.acc_density * k / np.sqrt(dt), (n, 3))
    imu_gyr = w_b + bg + rng.normal(0, cfg.gyr_density * k / np.sqrt(dt), (n, 3))

    markers = make_markers(cfg.marker_ids, cfg.marker_size)
    T_BC = T_BC_DEFAULT
    step = int(round(cfg.imu_rate / cfg.cam_rate))
    cam_idx = np.arange(0, n, step)
    obs_f, obs_i, obs_T, vo_f, vo_v = [], [], [], [], []
    n_vis = 0
    scale = rng.normal(0, cfg.vo_scale_sigma)
    for f, idx in enumerate(cam_idx):
        Twc = make_T(Rm[idx], p[idx]) @ T_BC
        c, Rwc = Twc[:3, 3], Twc[:3, :3]
        blackout = any(a <= t[idx] < b for a, b in cfg.blackouts)
        for mid, Twm in markers.items():
            m, nrm = Twm[:3, 3], Twm[:3, 2]
            pc = Rwc.T @ (m - c)
            d = np.linalg.norm(pc)
            if pc[2] <= 0.2 or not (0.3 < d < cfg.max_range):
                continue
            if np.arccos(pc[2] / d) > np.radians(cfg.fov_deg / 2):
                continue
            if nrm @ (c - m) / d < np.cos(np.radians(cfg.max_incidence_deg)):
                continue
            n_vis += 1
            if blackout or rng.random() < cfg.p_miss:
                continue
            obs_f.append(f); obs_i.append(mid)
            obs_T.append(_perturb(inv_T(Twc) @ Twm, d, cfg, rng))
        scale += rng.normal(0, 0.006 * np.sqrt(step * dt))
        vo_f.append(f)
        vo_v.append((1 + scale) * (Rm[idx].T @ vel[idx]) + rng.normal(0, cfg.vo_sigma, 3))

    meta = {"name": cfg.name, "simulated": True, "T_BC": T_BC.tolist(), "marker_size": cfg.marker_size,
            "vo_kind": "velocity", "start_pose": make_T(Rm[0], p[0]).tolist(), "sim_visible": n_vis,
            "sim_detected": len(obs_f), "blackouts": [list(b) for b in cfg.blackouts], "config": cfg.__dict__ | {"marker_ids": list(cfg.marker_ids), "blackouts": [list(b) for b in cfg.blackouts]},
            "imu_rate": cfg.imu_rate, "cam_rate": cfg.cam_rate}
    return Dataset(t, imu_acc, imu_gyr, t[cam_idx], np.array(obs_f, int), np.array(obs_i, int),
                   np.array(obs_T).reshape(-1, 4, 4), np.array(vo_f, int), np.array(vo_v), meta,
                   p[cam_idx], Rm[cam_idx], markers)
