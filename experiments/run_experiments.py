"""Runs the five experiments required by the handout over several random seeds.

  E1 simple path (L)            E2 marker-loss (12 s blackout)       E3 closed loop (2 laps)
  E4 relocalisation in a saved map                                     E5 difficult conditions

Methods compared (same recorded data replayed through each):
  IMU only | VIO | VIO + anchors (EKF with surveyed map) | VIO + landmarks (graph SLAM, map unknown)

Usage:  python experiments/run_experiments.py [--seeds 5] [--out results]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vislam.sim import simulate, SimConfig
from vislam.pipeline import run_filter, run_slam
from vislam.metrics import traj_metrics, map_metrics, detection_stats
from vislam.dataset import save_map
from vislam import viz

M_IMU, M_VIO, M_EKF, M_SLAM, M_REL = ("IMU only", "VIO", "VIO + anchors (EKF)", "VIO + landmarks (graph SLAM)",
                                      "Relocalisation (map + EKF)")

SCEN = {
    "E1_simple_path": dict(path="L", laps=1),
    "E2_marker_loss": dict(path="rect", laps=2, blackouts=((9.0, 21.0),)),
    "E3_closed_loop": dict(path="rect", laps=2),
    "E5_difficult": dict(path="rect", laps=2, speed=1.2, sway_deg=70, imu_noise_scale=2.0, det_noise_scale=2.0,
                         p_miss=0.5, p_outlier=0.03, marker_ids=(0, 2, 4, 6)),     # sparse + noisy + fast + false detections
}


def run_methods(ds, seed_dir=None):
    out, trajs = {}, {}
    trajs[M_IMU] = run_filter(ds, None, False)
    trajs[M_VIO] = run_filter(ds, None, True)
    trajs[M_EKF] = run_filter(ds, ds.map_gt, True)
    slam, mk, info = run_slam(ds)
    trajs[M_SLAM] = slam
    for k, tr in trajs.items():
        out[k] = traj_metrics(tr, ds) | {"runtime_s": tr.stats.get("runtime_s")}
    out[M_SLAM] |= map_metrics(mk, ds.map_gt) | {"graph_cost_before": info["cost_initial"], "graph_cost_after": info["cost_final"]}
    out[M_EKF]["gated_updates"] = trajs[M_EKF].stats["gated"]
    return out, trajs, mk


def summarise(per_seed):
    """mean +/- std of every metric over seeds."""
    res = {}
    for m in per_seed[0]:
        res[m] = {}
        for k in per_seed[0][m]:
            v = np.array([r[m].get(k, np.nan) for r in per_seed], float)
            if np.isfinite(v).any():
                res[m][k] = [float(np.nanmean(v)), float(np.nanstd(v))]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allres, saved_maps = {}, {}
    for name, kw in SCEN.items():
        per, first = [], None
        for s in range(a.seeds):
            ds = simulate(SimConfig(seed=s, name=name, **kw))
            r, trajs, mk = run_methods(ds)
            per.append(r)
            if name == "E3_closed_loop":
                saved_maps[s] = mk
            if s == 0:
                first = (ds, trajs, mk)
                if name == "E3_closed_loop":
                    save_map(f"{a.out}/map_closed_loop.json", mk, ds.meta["marker_size"])
                    ds.save("data/sim_closed_loop")          # representative dataset shipped with the repo
        allres[name] = {"summary": summarise(per), "dataset": detection_stats(first[0])}
        ds, trajs, mk = first
        viz.plot_scene(ds, trajs, mk, f"{a.out}/{name}_map.png", name.replace("_", " "))
        viz.plot_errors(ds, trajs, f"{a.out}/{name}_error.png", name.replace("_", " "), ds.meta["blackouts"])
        print(f"{name}: done")

    # ---------------- E4 relocalisation: new run, different start, previously built map
    per = []
    for s in range(a.seeds):
        mk = saved_maps[s]
        ds = simulate(SimConfig(seed=100 + s, path="reloc", name="E4_relocalisation", blackouts=((0.0, 3.0),)))
        nominal = np.array(simulate(SimConfig(seed=0, path="rect")).meta["start_pose"])   # user *assumes* the old start spot
        rel = run_filter(ds, mk, True, init_from_marker=True)
        base = run_filter(ds, None, True, init_pose=nominal)
        loc = run_filter(ds, mk, True, init_pose=nominal)
        r = {M_REL: traj_metrics(rel, ds) | {"reloc_time_s": rel.stats["reloc_time_s"], "runtime_s": rel.stats["runtime_s"]},
             M_VIO: traj_metrics(base, ds) | {"runtime_s": base.stats["runtime_s"]},
             "EKF + map, wrong init": traj_metrics(loc, ds)}
        per.append(r)
        if s == 0:
            tr0 = {M_VIO: base, M_REL: rel}; ds0 = ds
    allres["E4_relocalisation"] = {"summary": summarise(per), "dataset": detection_stats(ds0)}
    viz.plot_scene(ds0, tr0, saved_maps[0], f"{a.out}/E4_relocalisation_map.png", "E4 relocalisation (saved map)")
    viz.plot_errors(ds0, tr0, f"{a.out}/E4_relocalisation_error.png", "E4 relocalisation")
    json.dump(allres, open(f"{a.out}/metrics.json", "w"), indent=2)
    write_markdown(allres, a.seeds, f"{a.out}/RESULTS.md")
    print("wrote", a.out)


def write_markdown(allres, seeds, path):
    cols = [("ate_rmse_m", "ATE RMSE [m]"), ("rot_rmse_deg", "Rot RMSE [deg]"), ("endpoint_drift_m", "End drift [m]"),
            ("drift_per_m", "Drift/m"), ("loop_closure_residual_m", "Loop residual [m]"),
            ("marker_pos_err_mean_m", "Marker err [m]"), ("reloc_time_s", "Reloc time [s]")]
    L = [f"# Simulated experiment results (mean ± std over {seeds} seeds)\n"]
    for sc, d in allres.items():
        used = [c for c in cols if any(c[0] in m for m in d["summary"].values())]
        L += [f"## {sc}\n", f"Detection: {d['dataset']}\n",
              "| Method | " + " | ".join(c[1] for c in used) + " |", "|---|" + "---|" * len(used)]
        for m, v in d["summary"].items():
            L.append(f"| {m} | " + " | ".join(f"{v[c[0]][0]:.3f} ± {v[c[0]][1]:.3f}" if c[0] in v else "–" for c in used) + " |")
        L.append("")
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
