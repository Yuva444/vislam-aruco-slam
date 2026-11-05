"""Monocular visual odometry front-end (ORB-free KLT + essential matrix).

Produces the *direction* of camera translation (unit vector, body frame) between frames
`gap` apart.  Monocular vision cannot observe metric scale, so the EKF fuses it with the IMU
through `vo_direction`.  For metric VO use a library (ORB-SLAM3, OpenVINS, ARCore) and
write its body-frame velocity to vo.csv with meta['vo_kind']='velocity'.

NOTE: this module is validated only on synthetic geometry (see tests); tune on real footage.
"""
import cv2
import numpy as np


def direction_vo(video_path, K, dist, T_BC, gap=4, min_inliers=40, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames, f = {}, 0
    out_f, out_v = [], []
    R_BC = T_BC[:3, :3]
    prev = None
    buf = []
    while True:
        ok, img = cap.read()
        if not ok or (max_frames and f >= max_frames):
            break
        g = cv2.undistort(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), K, dist)
        buf.append(g)
        if len(buf) > gap + 1:
            buf.pop(0)
        if len(buf) == gap + 1:
            u = _pair_direction(buf[0], buf[-1], K, min_inliers)
            if u is not None:
                out_f.append(f)
                out_v.append(R_BC @ u)         # camera-frame direction -> body frame
        f += 1
    cap.release()
    return np.array(out_f, int), np.array(out_v).reshape(-1, 3)


def _pair_direction(g1, g2, K, min_inliers):
    p1 = cv2.goodFeaturesToTrack(g1, 400, 0.01, 10)
    if p1 is None or len(p1) < min_inliers:
        return None
    p2, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p1, None)
    good = st.ravel() == 1
    a, b = p1[good].reshape(-1, 2), p2[good].reshape(-1, 2)
    if len(a) < min_inliers:
        return None
    E, mask = cv2.findEssentialMat(a, b, K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3):
        return None
    n, R, t, mask = cv2.recoverPose(E, a, b, K, mask=mask)
    if n < min_inliers:
        return None
    return (-R.T @ t).ravel() / (np.linalg.norm(t) + 1e-12)   # camera-2 centre in camera-1 frame
