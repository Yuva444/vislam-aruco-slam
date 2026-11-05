"""Error-state EKF for visual-inertial navigation with fiducial pose updates.

Nominal state : p (W), v (W), R = R_WB, b_a, b_g
Error state   : dx = [dp, dv, dtheta, dba, dbg] (15),  R_true = R * Exp(dtheta)
IMU model     : f_m = R^T (a - g) + b_a + n_a ;  w_m = w + b_g + n_g
Updates       : (1) marker pose  (2) VO body velocity or direction
"""
from dataclasses import dataclass
import numpy as np
from .geometry import Exp, Log, skew, make_T, inv_T, GRAVITY


@dataclass
class EKFConfig:
    acc_n: float = 0.02
    gyr_n: float = 0.002
    acc_rw: float = 0.002
    gyr_rw: float = 2e-4
    vo_vel_sigma: float = 0.08       # m/s
    vo_dir_sigma: float = 0.15       # rad (direction-only VO)
    mk_pos0: float = 0.02            # marker pose noise: sigma = a + b * distance
    mk_pos1: float = 0.03
    mk_rot0_deg: float = 1.5
    mk_rot1_deg: float = 1.0
    map_sigma: float = 0.0           # extra position std of the marker map itself
    gate: bool = True
    gate_chi2: float = 22.46         # chi^2(6 dof) at 99.9 %
    p0_std: float = 0.02
    v0_std: float = 0.05
    th0_std: float = 0.02
    ba0_std: float = 0.1
    bg0_std: float = 0.01


class ESEKF:
    def __init__(self, T0, cfg=EKFConfig(), v0=None):
        self.cfg = cfg
        self.p = T0[:3, 3].copy()
        self.R = T0[:3, :3].copy()
        self.v = np.zeros(3) if v0 is None else np.asarray(v0, float)
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        c = cfg
        self.P = np.diag(np.r_[[c.p0_std] * 3, [c.v0_std] * 3, [c.th0_std] * 3, [c.ba0_std] * 3, [c.bg0_std] * 3]) ** 2
        self.n_acc = self.n_rej = 0

    def predict(self, acc, gyr, dt):
        c = self.cfg
        a_b, w = acc - self.ba, gyr - self.bg
        a_w = self.R @ a_b + GRAVITY
        dR = Exp(w * dt)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -self.R @ skew(a_b) * dt
        F[3:6, 9:12] = -self.R * dt
        F[6:9, 6:9] = dR.T
        F[6:9, 12:15] = -np.eye(3) * dt
        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt
        self.R = self.R @ dR
        Q = np.zeros(15)
        Q[3:6], Q[6:9] = c.acc_n ** 2 * dt, c.gyr_n ** 2 * dt
        Q[9:12], Q[12:15] = c.acc_rw ** 2 * dt, c.gyr_rw ** 2 * dt
        self.P = F @ self.P @ F.T + np.diag(Q)

    def _update(self, r, H, Rm, gate=False):
        S = H @ self.P @ H.T + Rm
        if gate and self.cfg.gate and r @ np.linalg.solve(S, r) > self.cfg.gate_chi2:
            self.n_rej += 1
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ r
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T
        self.p += dx[0:3]; self.v += dx[3:6]
        self.R = self.R @ Exp(dx[6:9])
        self.ba += dx[9:12]; self.bg += dx[12:15]
        self.n_acc += 1
        return True

    # ---- marker pose measurement:  T_WB_meas = T_WM * T_CM^-1 * T_BC^-1
    def marker_pose(self, T_WM, T_CM, T_BC):
        c = self.cfg
        T = T_WM @ inv_T(T_CM) @ inv_T(T_BC)
        d = np.linalg.norm(T_CM[:3, 3])
        sp = np.hypot(c.mk_pos0 + c.mk_pos1 * d, c.map_sigma)
        sr = np.radians(c.mk_rot0_deg + c.mk_rot1_deg * d)
        r = np.r_[T[:3, 3] - self.p, Log(self.R.T @ T[:3, :3])]
        H = np.zeros((6, 15)); H[0:3, 0:3] = np.eye(3); H[3:6, 6:9] = np.eye(3)
        return self._update(r, H, np.diag([sp ** 2] * 3 + [sr ** 2] * 3), gate=True)

    def reset_pose(self, T_WB, pos_std=0.1, rot_std=0.05, v_std=1.0):
        self.p, self.R = T_WB[:3, 3].copy(), T_WB[:3, :3].copy()
        self.v = np.zeros(3)
        self.P[:] = np.diag(np.r_[[pos_std] * 3, [v_std] * 3, [rot_std] * 3, [self.cfg.ba0_std] * 3, [self.cfg.bg0_std] * 3]) ** 2

    # ---- visual odometry: body-frame velocity  z = R^T v
    def vo_velocity(self, z):
        w = self.R.T @ self.v
        H = np.zeros((3, 15)); H[:, 3:6] = self.R.T; H[:, 6:9] = skew(w)
        return self._update(z - w, H, np.eye(3) * self.cfg.vo_vel_sigma ** 2)

    # ---- monocular VO: unit direction of body-frame velocity (scale unobservable from vision alone)
    def vo_direction(self, u):
        w = self.R.T @ self.v
        nw = np.linalg.norm(w)
        if nw < 0.15:
            return False
        h = w / nw
        Hw = (np.eye(3) - np.outer(h, h)) / nw
        H = np.zeros((3, 15)); H[:, 3:6] = Hw @ self.R.T; H[:, 6:9] = Hw @ skew(w)
        return self._update(u - h, H, np.eye(3) * self.cfg.vo_dir_sigma ** 2)

    @property
    def T(self):
        return make_T(self.R, self.p)
