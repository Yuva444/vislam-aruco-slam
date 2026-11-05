"""Synthetic end-to-end test of detection + PnP: render a marker, recover the known pose."""
import cv2
import numpy as np
from vislam.aruco import ArucoDetector
from vislam.geometry import make_T, Exp, rot_angle_deg


def render(T_CM, K, size, dictionary="DICT_4X4_50", mid=7, shape=(480, 640)):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    m = cv2.aruco.generateImageMarker(d, mid, 200)
    s = size / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    pts, _ = cv2.projectPoints(obj, cv2.Rodrigues(T_CM[:3, :3])[0], T_CM[:3, 3], K, None)
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [199, 0], [199, 199], [0, 199]]), pts.reshape(4, 2).astype(np.float32))
    img = np.full(shape, 255, np.uint8)
    warp = cv2.warpPerspective(m, H, shape[::-1], borderValue=255)
    mask = cv2.warpPerspective(np.full((200, 200), 255, np.uint8), H, shape[::-1])
    img[mask > 0] = warp[mask > 0]
    return img


def test_detect_and_pose():
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.0]])
    # marker facing the camera: marker y(up) = -camera y(down), marker z(out) = -camera z  ->  diag(1,-1,-1)
    T = make_T(Exp([0.1, -0.35, 0.05]) @ np.diag([1.0, -1, -1]), [0.1, -0.05, 1.5])
    det = ArucoDetector(K, None, 0.18)
    res = det.detect(render(T, K, 0.18))
    assert len(res) == 1 and res[0]["id"] == 7
    assert np.linalg.norm(res[0]["T_CM"][:3, 3] - T[:3, 3]) < 0.02
    assert rot_angle_deg(res[0]["T_CM"][:3, :3], T[:3, :3]) < 3.0
