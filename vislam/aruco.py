"""ArUco detection and camera-to-marker pose (T_CM) via planar PnP."""
import cv2
import numpy as np
from .geometry import T_from_rt


class ArucoDetector:
    def __init__(self, K, dist=None, marker_size=0.18, dictionary="DICT_4X4_50", max_reproj_px=3.0):
        self.K = np.asarray(K, float)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, float)
        self.max_reproj = max_reproj_px
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(d, p)
        s = marker_size / 2
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)  # TL,TR,BR,BL

    def detect(self, gray):
        """Return list of dict(id, T_CM, reproj_px, corners)."""
        corners, ids, _ = self.det.detectMarkers(gray)
        out = []
        for c, i in zip(corners, [] if ids is None else ids.ravel()):
            img = c.reshape(4, 2).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(self.obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(self.obj, rvec, tvec, self.K, self.dist)
            err = float(np.linalg.norm(proj.reshape(4, 2) - img, axis=1).mean())
            if err <= self.max_reproj:
                out.append({"id": int(i), "T_CM": T_from_rt(rvec, tvec), "reproj_px": err, "corners": img})
        return out

    def draw(self, frame, dets):
        for d in dets:
            cv2.polylines(frame, [d["corners"].astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(frame, f"id{d['id']} {np.linalg.norm(d['T_CM'][:3, 3]):.2f}m", tuple(d["corners"][0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame


def detect_video(video_path, detector, frame_times=None, annotated_out=None, max_frames=None):
    """Run detection over a video. Returns (cam_t, obs_frame, obs_id, obs_T)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer, t_list, fr, ids, Ts, f = None, [], [], [], [], 0
    while True:
        ok, img = cap.read()
        if not ok or (max_frames and f >= max_frames):
            break
        dets = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        for d in dets:
            fr.append(f); ids.append(d["id"]); Ts.append(d["T_CM"])
        if annotated_out:
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(annotated_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(detector.draw(img, dets))
        t_list.append(frame_times[f] if frame_times is not None else f / fps)
        f += 1
    cap.release()
    if writer:
        writer.release()
    return np.array(t_list), np.array(fr, int), np.array(ids, int), np.array(Ts).reshape(-1, 4, 4)
