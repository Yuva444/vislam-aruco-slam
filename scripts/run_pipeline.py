"""Process a dataset:  python scripts/run_pipeline.py --dataset data/sim_closed_loop --out results/run1
Optionally relocalise in a saved map:  --map results/map_closed_loop.json --reloc"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vislam.dataset import Dataset, save_map, load_map
from vislam.pipeline import run_filter, run_slam
from vislam.metrics import traj_metrics, map_metrics, detection_stats
from vislam.diagnostics import sensor_health
from vislam import viz

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--map", help="saved marker map (JSON) used as anchors / for relocalisation")
ap.add_argument("--reloc", action="store_true", help="pose unknown at start: relocalise from first marker")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ds = Dataset.load(a.dataset)
amap = load_map(a.map) if a.map else ds.map_gt
trajs = {"IMU only": run_filter(ds, None, False), "VIO": run_filter(ds, None, True)}
est_map = None
if amap is not None:
    trajs["VIO + anchors (EKF)" if not a.reloc else "Relocalisation (map + EKF)"] = run_filter(ds, amap, True, init_from_marker=a.reloc)
slam, est_map, info = run_slam(ds)
trajs["VIO + landmarks (graph SLAM)"] = slam
save_map(f"{a.out}/map_estimated.json", est_map, ds.meta.get("marker_size", 0))
report = {"sensor_health": sensor_health(ds), "detections": detection_stats(ds), "graph": info, "methods": {}}
for k, tr in trajs.items():
    report["methods"][k] = traj_metrics(tr, ds) | tr.stats
    np.savetxt(f"{a.out}/traj_{re.sub(r'[^a-z0-9]+', '_', k.lower()).strip('_')}.csv", np.column_stack([ds.cam_t, tr.p]), delimiter=",", header="t,x,y,z", comments="")
if ds.map_gt:
    report["map"] = map_metrics(est_map, ds.map_gt)
json.dump(report, open(f"{a.out}/report.json", "w"), indent=2, default=float)
if ds.gt_p is not None:
    viz.plot_scene(ds, trajs, est_map, f"{a.out}/map.png", ds.meta.get("name", ""))
    viz.plot_errors(ds, trajs, f"{a.out}/error.png")
print(json.dumps({k: {m: round(v, 3) for m, v in d.items() if isinstance(v, float)} for k, d in report["methods"].items()}, indent=1))
