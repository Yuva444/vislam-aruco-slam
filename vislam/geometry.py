"""Rigid-body helpers.  Convention: T_AB maps coordinates expressed in frame B into frame A.

Frames
  W : world / room frame, z up (gravity = -z)
  B : IMU/body frame, FLU (x forward, y left, z up)
  C : camera frame, OpenCV (x right, y down, z forward)
  M : marker frame, x right, y up, z out of the marker face
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

GRAVITY = np.array([0.0, 0.0, -9.81])


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0.0]])


def Exp(w):
    """SO(3) exponential (Rodrigues)."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3) + skew(w)
    K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def Log(R):
    return Rot.from_matrix(R).as_rotvec()


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    return make_T(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def T_from_rt(rvec, tvec):
    return make_T(Exp(np.asarray(rvec, float).ravel()), np.asarray(tvec, float).ravel())


def rot_angle_deg(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Log(Ra.T @ Rb))))


# Camera axes expressed in the body frame: cam-z -> body-x, cam-x -> body -y, cam-y -> body -z.
R_BC_DEFAULT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
T_BC_DEFAULT = make_T(R_BC_DEFAULT, np.zeros(3))


def gravity_alignment(acc_mean):
    """R_WB (yaw = 0) from a stationary accelerometer mean (specific force, body frame).

    Rows of R_WB are the world axes expressed in the body frame."""
    z = acc_mean / np.linalg.norm(acc_mean)          # world up, in body coordinates
    x = np.array([1.0, 0, 0]) - z * z[0]             # horizontal projection of body-x
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])
