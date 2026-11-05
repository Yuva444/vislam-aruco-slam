"""Calibrate from checkerboard photos:  python scripts/calibrate_camera.py "calibration/img/*.jpg" --board 9 6 --square-mm 25"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vislam.calibration import calibrate_from_images

ap = argparse.ArgumentParser()
ap.add_argument("images")
ap.add_argument("--board", type=int, nargs=2, default=[9, 6], help="INNER corners (cols rows)")
ap.add_argument("--square-mm", type=float, default=25.0)
ap.add_argument("--out", default="calibration/camera.json")
a = ap.parse_args()
r = calibrate_from_images(a.images, tuple(a.board), a.square_mm / 1000, a.out)
print(f"{r['n_images']} images | RMS reprojection error {r['rms_reproj_px']:.3f} px (target < 0.5 px)\nK={r['K']}\ndist={r['dist']}")
