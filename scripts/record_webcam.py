"""Record a laptop/USB webcam to MP4 + per-frame monotonic timestamps (same clock family as an IMU logger
that you start on the same machine).   python scripts/record_webcam.py --out data/raw/run1 --seconds 60"""
import argparse, os, time
import cv2


def record(out, seconds=60, cam=0, width=1280, height=720, show=False):
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # fixed focus keeps intrinsics valid
    w, h = int(cap.get(3)), int(cap.get(4))
    vw = cv2.VideoWriter(f"{out}/video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    t0, ts = time.monotonic(), []
    while time.monotonic() - t0 < seconds:
        ok, img = cap.read()
        if not ok:
            break
        ts.append(time.monotonic() - t0)
        vw.write(img)
        if show:
            cv2.imshow("recording (q to stop)", img)
            if cv2.waitKey(1) == 113:
                break
    cap.release(); vw.release()
    with open(f"{out}/video_timestamps.csv", "w") as f:
        f.write("frame,t\n" + "\n".join(f"{i},{t:.6f}" for i, t in enumerate(ts)))
    return len(ts), (ts[-1] if ts else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--cam", type=int, default=0); ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    print("frames, duration:", record(a.out, a.seconds, a.cam, show=a.show))
