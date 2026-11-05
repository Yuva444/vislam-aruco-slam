"""Create a simulated dataset:  python scripts/simulate_dataset.py --path rect --laps 2 --out data/sim_demo"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vislam.sim import simulate, SimConfig

ap = argparse.ArgumentParser()
ap.add_argument("--path", default="rect", choices=["L", "rect", "reloc"])
ap.add_argument("--laps", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--blackout", type=float, nargs=2, action="append", metavar=("T0", "T1"))
ap.add_argument("--out", required=True)
a = ap.parse_args()
ds = simulate(SimConfig(path=a.path, laps=a.laps, seed=a.seed, blackouts=tuple(map(tuple, a.blackout or ()))))
ds.save(a.out)
print(f"saved {a.out}: {ds.imu_t[-1]:.1f}s, {len(ds.cam_t)} frames, {len(ds.obs_id)} marker detections")
