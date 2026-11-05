"""Dataset container + human-readable on-disk format (CSV + JSON) so runs can be replayed.

Folder layout
  imu.csv        t,ax,ay,az,gx,gy,gz          (specific force m/s^2, rate rad/s, body frame)
  cam_times.csv  frame,t
  markers.csv    frame,t,id,tx,ty,tz,rx,ry,rz  (T_CM: marker pose in camera frame, rotvec)
  vo.csv         frame,vx,vy,vz                (visual-odometry body velocity OR unit direction)
  gt.csv         t,px,py,pz,qx,qy,qz,qw        (optional ground truth, world frame)
  map_gt.json    surveyed marker poses         (optional)
  meta.json      calibration, extrinsics, vo_kind, notes ...
"""
import json
import os
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from .geometry import T_from_rt, T_BC_DEFAULT


def _w(path, header, arr):
    arr = np.asarray(arr)
    np.savetxt(path, arr.reshape(len(arr), -1), delimiter=",", header=header, comments="", fmt="%.9g")


def _r(path, ncol):
    if not os.path.exists(path):
        return np.zeros((0, ncol))
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return a if a.size else np.zeros((0, ncol))


@dataclass
class Dataset:
    imu_t: np.ndarray
    imu_acc: np.ndarray
    imu_gyr: np.ndarray
    cam_t: np.ndarray
    obs_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    obs_id: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    obs_T: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4)))
    vo_frame: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    vo_vec: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    meta: dict = field(default_factory=dict)
    gt_p: np.ndarray = None
    gt_R: np.ndarray = None
    map_gt: dict = None

    @property
    def T_BC(self):
        return np.array(self.meta.get("T_BC", T_BC_DEFAULT.tolist()))

    def obs_by_frame(self):
        out = {}
        for f, i, T in zip(self.obs_frame, self.obs_id, self.obs_T):
            out.setdefault(int(f), []).append((int(i), T))
        return out

    def vo_by_frame(self):
        return {int(f): v for f, v in zip(self.vo_frame, self.vo_vec)}

    # ------------------------------------------------------------------ I/O
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        _w(f"{folder}/imu.csv", "t,ax,ay,az,gx,gy,gz", np.column_stack([self.imu_t, self.imu_acc, self.imu_gyr]))
        _w(f"{folder}/cam_times.csv", "frame,t", np.column_stack([np.arange(len(self.cam_t)), self.cam_t]))
        rows = []
        for f, i, T in zip(self.obs_frame, self.obs_id, self.obs_T):
            rv = Rot.from_matrix(T[:3, :3]).as_rotvec()
            rows.append([f, self.cam_t[int(f)], i, *T[:3, 3], *rv])
        _w(f"{folder}/markers.csv", "frame,t,id,tx,ty,tz,rx,ry,rz", np.array(rows).reshape(-1, 9))
        _w(f"{folder}/vo.csv", "frame,vx,vy,vz", np.column_stack([self.vo_frame, self.vo_vec]).reshape(-1, 4))
        if self.gt_p is not None:
            q = Rot.from_matrix(self.gt_R).as_quat()
            _w(f"{folder}/gt.csv", "t,px,py,pz,qx,qy,qz,qw", np.column_stack([self.cam_t, self.gt_p, q]))
        if self.map_gt is not None:
            save_map(f"{folder}/map_gt.json", self.map_gt, self.meta.get("marker_size", 0.0))
        json.dump(self.meta, open(f"{folder}/meta.json", "w"), indent=2)

    @staticmethod
    def load(folder):
        imu = _r(f"{folder}/imu.csv", 7)
        cam = _r(f"{folder}/cam_times.csv", 2)
        mk = _r(f"{folder}/markers.csv", 9)
        vo = _r(f"{folder}/vo.csv", 4)
        meta = json.load(open(f"{folder}/meta.json")) if os.path.exists(f"{folder}/meta.json") else {}
        ds = Dataset(imu[:, 0], imu[:, 1:4], imu[:, 4:7], cam[:, 1],
                     mk[:, 0].astype(int), mk[:, 2].astype(int),
                     np.array([T_from_rt(r[6:9], r[3:6]) for r in mk]).reshape(-1, 4, 4),
                     vo[:, 0].astype(int), vo[:, 1:4], meta)
        gt = _r(f"{folder}/gt.csv", 8)
        if len(gt):
            ds.gt_p = gt[:, 1:4]
            ds.gt_R = Rot.from_quat(gt[:, 4:8]).as_matrix()
        if os.path.exists(f"{folder}/map_gt.json"):
            ds.map_gt = load_map(f"{folder}/map_gt.json")
        return ds


def save_map(path, marker_map, marker_size=0.0):
    out = {"marker_size": marker_size, "markers": {}}
    for i, T in marker_map.items():
        q = Rot.from_matrix(T[:3, :3]).as_quat().tolist()
        out["markers"][str(i)] = {"T_WM": np.asarray(T).tolist(), "position": T[:3, 3].tolist(), "quat_xyzw": q}
    json.dump(out, open(path, "w"), indent=2)


def load_map(path):
    d = json.load(open(path))
    return {int(i): np.array(v["T_WM"]) for i, v in d["markers"].items()}
