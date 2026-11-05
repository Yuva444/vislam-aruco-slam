"""Plain-language analysis of a finished run (used by the dashboard's results page)."""
import numpy as np
from .metrics import traj_metrics, detection_stats

PREFERRED = ("VIO + anchors (EKF)", "VIO + landmarks (graph SLAM)", "VIO")


def method_table(ds, trajs):
    """{method: metrics dict} (ATE, rotation error, drift, runtime ...)."""
    return {n: traj_metrics(tr, ds) | {"runtime_s": tr.stats.get("runtime_s")} for n, tr in trajs.items()}


def gap_stats(rp):
    """Error behaviour inside vs. outside the no-marker-visible intervals of a Replay, per method."""
    out = {}
    if rp.gt_p is None:
        return out
    t = rp.t
    in_gap = np.zeros(len(t), bool)
    for a, b in rp.spans:
        in_gap |= (t >= a) & (t <= b)
    for n, m in rp.methods.items():
        e = m["err"]
        ok = np.isfinite(e)
        d = {"mean_in_gap": float(e[ok & in_gap].mean()) if (ok & in_gap).any() else None,
             "mean_with_markers": float(e[ok & ~in_gap].mean()) if (ok & ~in_gap).any() else None,
             "worst_in_gap": float(e[ok & in_gap].max()) if (ok & in_gap).any() else None, "gaps": []}
        for a, b in rp.spans:
            k0, k1 = int(np.searchsorted(t, a)), int(min(np.searchsorted(t, b), len(t) - 1))
            k2 = int(min(np.searchsorted(t, b + 2.0), len(t) - 1))
            d["gaps"].append(dict(start=float(a), end=float(b), err_start=float(e[k0]), err_end=float(e[k1]), err_after=float(e[k2])))
        out[n] = d
    return out


def narrative(ds, rp, table):
    """A few plain-English sentences summarising the run (markdown)."""
    L = []
    ate = {n: m["ate_rmse_m"] for n, m in table.items() if m}
    if not ate or rp.gt_p is None:
        det = detection_stats(ds)
        return [f"Markers were detected in **{det['frames_with_marker_pct']:.0f} %** of camera frames "
                f"(longest marker-free stretch **{det['longest_marker_free_gap_s']:.1f} s**). "
                "This dataset has no ground truth, so position errors cannot be computed."]
    best = min(ate, key=ate.get)
    plen = table[best].get("path_len_m", float("nan"))
    L.append(f"**Best estimate: {best}** — on average {ate[best] * 100:.1f} cm from the true position over a {plen:.1f} m walk.")
    if "IMU only" in ate and best != "IMU only":
        L.append(f"Inertial sensors alone drift away quickly: **{ate['IMU only']:.2f} m** average error "
                 f"(about {ate['IMU only'] / ate[best]:.0f}× worse than {best}). Adding the camera and markers is what keeps the estimate usable.")
    det = detection_stats(ds)
    L.append(f"Markers were detected in **{det['frames_with_marker_pct']:.0f} %** of camera frames; "
             f"the longest stretch without any marker lasted **{det['longest_marker_free_gap_s']:.1f} s**.")
    gs = gap_stats(rp)
    ref = next((m for m in PREFERRED if m in gs and gs[m]["gaps"]), None)
    if ref:
        g = max(gs[ref]["gaps"], key=lambda x: x["end"] - x["start"])
        verb = "grew" if g["err_end"] > g["err_start"] else "stayed about the same"
        txt = (f"During the longest marker-free stretch ({g['start']:.0f}–{g['end']:.0f} s), the error of **{ref}** "
               f"{verb} from {g['err_start'] * 100:.1f} cm to {g['err_end'] * 100:.1f} cm; "
               f"two seconds after a marker came back it was {g['err_after'] * 100:.1f} cm.")
        if "IMU only" in gs and gs["IMU only"]["worst_in_gap"]:
            txt += f" Inertial-only error reached {gs['IMU only']['worst_in_gap']:.1f} m in the same period."
        L.append(txt)
    elif det["longest_marker_free_gap_s"] < 1.0:
        L.append("No stretch without markers lasted longer than 1 s, so the effect of losing the markers cannot be seen on this path "
                 "— tick *Hide all markers for a while* to test it.")
    if "VIO + landmarks (graph SLAM)" in ate and "VIO" in ate:
        d = ate["VIO"] - ate["VIO + landmarks (graph SLAM)"]
        L.append("Graph SLAM (markers with *unknown* positions, mapped on the fly) " +
                 (f"improved on plain VIO by {d * 100:.1f} cm." if d > 0.005 else
                  "was about as accurate as plain VIO on this path — it mostly pays off when the path closes a loop."))
    return L
