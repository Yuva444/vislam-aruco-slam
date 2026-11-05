"""Print-ready ArUco markers:  python scripts/generate_markers.py --ids 0-7 --size-mm 180 --out markers/
Print at 100 % scale (no 'fit to page') and measure the black square with a ruler -> pass it as --marker-size."""
import argparse, os
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ids", default="0-7")
ap.add_argument("--size-mm", type=float, default=180)
ap.add_argument("--dict", default="DICT_4X4_50")
ap.add_argument("--out", default="markers")
a = ap.parse_args()
lo, hi = map(int, a.ids.split("-"))
os.makedirs(a.out, exist_ok=True)
d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dict))
dpi = 300
px = int(a.size_mm / 25.4 * dpi)
for i in range(lo, hi + 1):
    m = cv2.aruco.generateImageMarker(d, i, px)
    pad = px // 4
    m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)   # white quiet zone
    cv2.putText(m, f"ID {i}  ({a.size_mm:.0f} mm)", (pad, pad // 2), cv2.FONT_HERSHEY_SIMPLEX, px / 400, 0, 2)
    cv2.imwrite(f"{a.out}/aruco_{i:02d}.png", m)
print(f"wrote {hi - lo + 1} markers to {a.out}/")
