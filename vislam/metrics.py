"""Quantitative evaluation (handout section 9)."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def align_se3(est, gt):
    """Rigid (no scale) Umeyama alignment of est -> gt; returns R, t."""
    me, mg = est.mean(0), gt.mean(0)
    U, _, Vt = np.linalg.svd((gt - mg).T @ (est - me))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, mg - R @ me


def traj_metrics(traj, ds):
    ok = traj.ok
    if ok.sum() < 3 or ds.gt_p is None:
        return {}
    e = np.linalg.norm(traj.p[ok] - ds.gt_p[ok], axis=1)
    R, t = align_se3(traj.p[ok], ds.gt_p[ok])
    e_al = np.linalg.norm((traj.p[ok] @ R.T + t) - ds.gt_p[ok], axis=1)
    rot = (Rot.from_matrix(traj.R[ok]).inv() * Rot.from_matrix(ds.gt_R[ok])).magnitude()
    plen = float(np.linalg.norm(np.diff(ds.gt_p, axis=0), axis=1).sum())
    out = {"ate_rmse_m": float(np.sqrt((e ** 2).mean())), "ate_max_m": float(e.max()),
           "ate_aligned_rmse_m": float(np.sqrt((e_al ** 2).mean())),
           "rot_rmse_deg": float(np.degrees(np.sqrt((rot ** 2).mean()))),
           "endpoint_drift_m": float(e[-1]), "path_len_m": plen, "drift_per_m": float(e[-1] / plen)}
    if np.linalg.norm(ds.gt_p[-1] - ds.gt_p[0]) < 0.3:   # closed loop in the ground truth
        out["loop_closure_residual_m"] = float(np.linalg.norm(traj.p[ok][-1] - traj.p[ok][0]))
    return out


def map_metrics(est_map, gt_map):
    ids = sorted(set(est_map) & set(gt_map))
    if not ids:
        return {}
    pe = [np.linalg.norm(est_map[i][:3, 3] - gt_map[i][:3, 3]) for i in ids]
    re = [np.degrees(np.linalg.norm(Rot.from_matrix(est_map[i][:3, :3].T @ gt_map[i][:3, :3]).as_rotvec())) for i in ids]
    return {"marker_pos_err_mean_m": float(np.mean(pe)), "marker_pos_err_max_m": float(np.max(pe)),
            "marker_rot_err_mean_deg": float(np.mean(re)), "n_markers_mapped": len(ids)}


def detection_stats(ds):
    fr = np.unique(ds.obs_frame)
    nF = len(ds.cam_t)
    mask = np.zeros(nF, bool); mask[fr.astype(int)] = True
    # longest marker-free interval (seconds) between consecutive detections
    t = ds.cam_t[mask]
    gap = float(np.max(np.diff(np.r_[ds.cam_t[0], t, ds.cam_t[-1]]))) if len(t) else float(ds.cam_t[-1] - ds.cam_t[0])
    out = {"frames_with_marker_pct": float(100 * mask.mean()), "longest_marker_free_gap_s": gap,
           "n_detections": int(len(ds.obs_id))}
    if "sim_visible" in ds.meta and ds.meta["sim_visible"]:
        out["detection_success_pct"] = float(100 * ds.meta["sim_detected"] / ds.meta["sim_visible"])
    return out
