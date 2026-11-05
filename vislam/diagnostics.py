"""Sensor-stream health report shown in the dashboard / CLI (rates, jitter, drops, sync overlap)."""
import numpy as np


def sensor_health(ds):
    di, dc = np.diff(ds.imu_t), np.diff(ds.cam_t)
    mi, mc = np.median(di), np.median(dc)
    ids, cnt = np.unique(ds.obs_id, return_counts=True)
    return {
        "imu_rate_hz": float(1 / mi), "imu_jitter_ms": float(1e3 * di.std()),
        "imu_dropped_samples": int((di > 1.5 * mi).sum()),
        "cam_rate_hz": float(1 / mc), "cam_dropped_frames": int((dc > 1.5 * mc).sum()),
        "duration_s": float(ds.imu_t[-1] - ds.imu_t[0]),
        "stream_overlap_s": float(min(ds.imu_t[-1], ds.cam_t[-1]) - max(ds.imu_t[0], ds.cam_t[0])),
        "cam_imu_time_offset_s": float(ds.meta.get("cam_imu_offset_s", 0.0)),
        "accel_norm_mean": float(np.linalg.norm(ds.imu_acc, axis=1).mean()),   # ~9.81 when static => units OK
        "detections_per_id": {int(i): int(c) for i, c in zip(ids, cnt)},
    }
