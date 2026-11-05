"""Matplotlib figures (used by experiments and CLI)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"IMU only": "tab:red", "VIO": "tab:orange", "VIO + anchors (EKF)": "tab:green",
       "VIO + landmarks (graph SLAM)": "tab:blue", "Relocalisation (map + EKF)": "tab:purple", "Ground truth": "k"}


def plot_scene(ds, trajs, est_map=None, path=None, title="", room=(6, 5), ax=None, clip=None):
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, room[0], room[0], 0, 0], [0, 0, room[1], room[1], 0], color="0.5", lw=2)
    if ds.gt_p is not None:
        ax.plot(ds.gt_p[:, 0], ds.gt_p[:, 1], "k--", lw=1.5, label="Ground truth")
    for name, tr in trajs.items():
        p = tr.p[tr.ok]
        if clip:
            p = p[np.all(np.abs(p[:, :2] - np.array([3, 2.5])) < clip, 1)]
        ax.plot(p[:, 0], p[:, 1], lw=1.6, color=COL.get(name), label=name)
    if ds.map_gt:
        for i, T in ds.map_gt.items():
            ax.plot(*T[:2, 3], "ks", ms=7, label="Marker (surveyed)" if i == min(ds.map_gt) else None)
            ax.annotate(str(i), T[:2, 3] + 0.1, fontsize=8)
    if est_map:
        for i, T in est_map.items():
            ax.plot(*T[:2, 3], "bx", ms=8, mew=2, label="Marker (estimated)" if i == min(est_map) else None)
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(title)
    ax.set_xlim(-0.8, room[0] + 0.8); ax.set_ylim(-0.8, room[1] + 0.8)
    if own:
        ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=130)
        plt.close()
    return ax


def plot_errors(ds, trajs, path=None, title="", blackouts=()):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for name, tr in trajs.items():
        e = np.where(tr.ok, np.linalg.norm(tr.p - ds.gt_p, axis=1), np.nan)
        ax.plot(ds.cam_t, e, color=COL.get(name), label=name)
    for a, b in blackouts:
        ax.axvspan(a, b, color="0.85", label="no marker visible")
    ax.set_yscale("log"); ax.set_xlabel("time [s]"); ax.set_ylabel("position error [m]"); ax.grid(alpha=.3, which="both")
    ax.set_title(title); ax.legend(fontsize=7); plt.tight_layout()
    if path:
        plt.savefig(path, dpi=130)
    plt.close(fig)
