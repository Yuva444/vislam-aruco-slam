"""Camera intrinsic calibration (checkerboard) and camera-IMU time-offset estimation."""
import glob
import json
import cv2
import numpy as np


def calibrate_from_images(pattern_glob, board=(9, 6), square_m=0.025, out_json=None):
    """Zhang calibration. `board` = inner corners (cols, rows). Returns dict with K, dist, rms, per-image error."""
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square_m
    obj, img, used, size = [], [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    for fn in sorted(glob.glob(pattern_glob)):
        g = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        size = g.shape[::-1]
        ok, c = cv2.findChessboardCorners(g, board)
        if ok:
            obj.append(objp); img.append(cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit)); used.append(fn)
    if len(obj) < 8:
        raise RuntimeError(f"Only {len(obj)} usable boards found; need >= 8 (aim for 15-25 at varied poses).")
    rms, K, dist, rv, tv = cv2.calibrateCamera(obj, img, size, None, None)
    per = []
    for o, i, r, t in zip(obj, img, rv, tv):
        p, _ = cv2.projectPoints(o, r, t, K, dist)
        per.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, 1)))))
    res = {"K": K.tolist(), "dist": dist.ravel().tolist(), "image_size": list(size), "rms_reproj_px": float(rms),
           "per_image_px": per, "n_images": len(used), "board": list(board), "square_m": square_m}
    if out_json:
        json.dump(res, open(out_json, "w"), indent=2)
    return res


def angular_rate_from_rotations(t, R):
    """|omega| (rad/s) from a sequence of rotation matrices (e.g. marker/PnP or VO orientation)."""
    from scipy.spatial.transform import Rotation as Rot
    r = Rot.from_matrix(R)
    w = (r[:-1].inv() * r[1:]).magnitude() / np.diff(t)
    return 0.5 * (t[:-1] + t[1:]), w


def estimate_time_offset(t_imu, w_imu_norm, t_cam, w_cam_norm, max_lag=0.5, rate=100.0):
    """Offset d (s) maximising corr(w_cam(t), w_imu(t + d)).  Camera timestamps should be shifted by +d.

    Shake the rig (rotate quickly about all axes) at the start of a recording for a well-conditioned peak."""
    lo = max(t_imu[0], t_cam[0]) + max_lag
    hi = min(t_imu[-1], t_cam[-1]) - max_lag
    grid = np.arange(lo, hi, 1 / rate)
    a = np.interp(grid, t_cam, w_cam_norm)
    a = (a - a.mean()) / (a.std() + 1e-12)
    lags = np.arange(-max_lag, max_lag, 1 / rate)
    score = []
    for d in lags:
        b = np.interp(grid + d, t_imu, w_imu_norm)
        score.append(np.mean(a * (b - b.mean()) / (b.std() + 1e-12)))
    j = int(np.argmax(score))
    return float(lags[j]), float(score[j])
