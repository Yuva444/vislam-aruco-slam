"""Animated replay (MP4 + interactive HTML) of a dataset processed by one or more methods.

  python scripts/make_replay.py --dataset data/sim_closed_loop --out results/replay.mp4
  python scripts/make_replay.py --simulate E2 --out docs/replay_E2.mp4 --html docs/replay_E2.html
  python scripts/make_replay.py --dataset data/room_run1 --methods "VIO" "VIO + landmarks (graph SLAM)" --speed 2
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vislam.dataset import Dataset, load_map
from vislam.pipeline import METHODS, run_method
from vislam.replay import build_replay, plotly_replay, render_video
from vislam.sim import simulate, SimConfig

# same scenario definitions as experiments/run_experiments.py (E2 has the 12 s marker blackout)
PRESETS = {"E1": dict(path="L", laps=1), "E2": dict(path="rect", laps=2, blackouts=((9.0, 21.0),)),
           "E3": dict(path="rect", laps=2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", help="dataset folder (imu.csv, markers.csv, ...)")
    g.add_argument("--simulate", choices=sorted(PRESETS), help="generate a simulated scenario instead")
    ap.add_argument("--methods", nargs="+", default=["IMU only", "VIO", "VIO + anchors (EKF)", "VIO + landmarks (graph SLAM)"],
                    choices=METHODS)
    ap.add_argument("--map", help="saved marker map JSON for the anchor / relocalisation methods (default: dataset map)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/replay.mp4", help="video file (.mp4, or .gif)")
    ap.add_argument("--html", help="also write a standalone interactive HTML replay (slider + play button)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed vs. real time")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--trail", type=float, default=4.0, help="trail length [s]")
    ap.add_argument("--min-gap", type=float, default=1.0, help="shortest marker-free interval to highlight [s]")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()

    ds = Dataset.load(a.dataset) if a.dataset else simulate(SimConfig(seed=a.seed, name=a.simulate + "_replay", **PRESETS[a.simulate]))
    amap = load_map(a.map) if a.map else None
    trajs, est_map = {}, None
    for m in a.methods:
        t0 = time.perf_counter()
        trajs[m], mk, _ = run_method(ds, m, amap)
        est_map = mk or est_map
        print(f"{m}: {time.perf_counter() - t0:.1f} s")
    rp = build_replay(ds, trajs, est_map, a.min_gap, name=ds.meta.get("name", ""))
    print(f"no-marker intervals: {[(round(x, 1), round(y, 1)) for x, y in rp.spans]}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.html:
        plotly_replay(rp, speed=a.speed, trail_s=a.trail).write_html(a.html, include_plotlyjs="cdn", auto_play=False)
        print("wrote", a.html)
    if not a.no_video:
        last = [0]

        def prog(x):
            if x - last[0] >= 0.1 or x == 1:
                last[0] = x; print(f"  rendering {x:4.0%}", end="\r", flush=True)
        out = render_video(rp, a.out, fps=a.fps, speed=a.speed, trail_s=a.trail, progress=prog)
        print("\nwrote", out)


if __name__ == "__main__":
    main()
