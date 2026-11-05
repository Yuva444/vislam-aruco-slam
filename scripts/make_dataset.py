"""Turn a REAL recording (video + IMU CSV) into a replayable dataset.

  python scripts/make_dataset.py --video raw/run1/video.mp4 --imu raw/run1/imu.csv \
      --calib calibration/camera.json --marker-size 0.18 --out data/room_run1 \
      [--video-ts raw/run1/video_timestamps.csv] [--imu-time-unit ms] [--vo direction] [--time-offset 0.0]

IMU CSV needs columns (names configurable): time, ax ay az (m/s^2, specific force), gx gy gz (rad/s).
Axes must follow the body convention x-forward / y-left / z-up; if your phone differs, set --imu-axes.
Ground truth (optional): --gt gt.csv with t,x,y,z (tape-measured waypoints are fine) and --map-gt map_gt.json."""
import argparse, json, os, shutil, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vislam.aruco import ArucoDetector, detect_video
from vislam.dataset import Dataset, load_map
from vislam.geometry import T_BC_DEFAULT, R_BC_DEFAULT, make_T
from vislam.vo import direction_vo

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--imu", required=True)
ap.add_argument("--calib", required=True); ap.add_argument("--marker-size", type=float, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--video-ts")
ap.add_argument("--imu-time-unit", default="s", choices=["s", "ms", "ns"])
ap.add_argument("--imu-cols", nargs=7, default=["time", "ax", "ay", "az", "gx", "gy", "gz"])
ap.add_argument("--imu-axes", nargs=3, type=float, default=[1, 1, 1], help="sign flips applied to x y z of both sensors")
ap.add_argument("--time-offset", type=float, default=0.0, help="seconds added to camera time (from estimate_time_offset)")
ap.add_argument("--dict", default="DICT_4X4_50"); ap.add_argument("--vo", choices=["none", "direction"], default="direction")
ap.add_argument("--R-BC", type=float, nargs=9, help="camera->body rotation, row-major (default: camera looks along body x)")
ap.add_argument("--gt"); ap.add_argument("--map-gt"); ap.add_argument("--annotate", action="store_true")
a = ap.parse_args()

cal = json.load(open(a.calib)); K, dist = np.array(cal["K"]), np.array(cal["dist"])
T_BC = make_T(np.array(a.R_BC).reshape(3, 3), np.zeros(3)) if a.R_BC else T_BC_DEFAULT
imu = pd.read_csv(a.imu); c = a.imu_cols
scale = {"s": 1.0, "ms": 1e-3, "ns": 1e-9}[a.imu_time_unit]
t_imu = imu[c[0]].to_numpy(float) * scale
sg = np.array(a.imu_axes)
acc, gyr = imu[c[1:4]].to_numpy(float) * sg, imu[c[4:7]].to_numpy(float) * sg
ft = pd.read_csv(a.video_ts)["t"].to_numpy(float) if a.video_ts else None
det = ArucoDetector(K, dist, a.marker_size, a.dict)
os.makedirs(a.out, exist_ok=True)
cam_t, fr, ids, Ts = detect_video(a.video, det, ft, f"{a.out}/annotated.mp4" if a.annotate else None)
cam_t = cam_t + a.time_offset
t0 = max(t_imu[0], cam_t[0])                      # common time origin
keep = t_imu >= t0
vo_f, vo_v = (np.zeros(0, int), np.zeros((0, 3)))
if a.vo == "direction":
    vo_f, vo_v = direction_vo(a.video, K, dist, T_BC)
meta = {"name": os.path.basename(a.out), "simulated": False, "T_BC": T_BC.tolist(), "marker_size": a.marker_size,
        "vo_kind": "direction", "cam_imu_offset_s": a.time_offset, "K": K.tolist(), "dist": dist.tolist(),
        "imu_rate": float(1 / np.median(np.diff(t_imu)))}
ds = Dataset(t_imu[keep] - t0, acc[keep], gyr[keep], cam_t - t0, fr, ids, Ts, vo_f, vo_v, meta)
if a.map_gt:
    ds.map_gt = load_map(a.map_gt)
if a.gt:
    g = pd.read_csv(a.gt); gp = g[["x", "y", "z"]].to_numpy(float)
    ds.gt_p = np.column_stack([np.interp(ds.cam_t, g["t"] - t0, gp[:, j]) for j in range(3)])
    ds.gt_R = np.tile(np.eye(3), (len(ds.cam_t), 1, 1))     # heading GT not provided -> rotation metrics invalid
ds.save(a.out)
print(f"{a.out}: {len(ds.cam_t)} frames, {len(ids)} detections, {len(ds.imu_t)} IMU samples, {len(vo_f)} VO rows")
