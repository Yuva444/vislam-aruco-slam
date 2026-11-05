"""Animated replay of a run: true position vs. every method's estimate, camera view direction,
marker sightings and a live error-versus-time readout.

Two renderers share one data container (`Replay`):
  plotly_replay(rp)   interactive figure with a time slider + play button   (dashboard / standalone HTML)
  render_video(rp)    MP4 (or GIF fallback) using Matplotlib                 (reports, slides)

What is drawn
  black dot            true position (ground truth) with a fading black trail
  coloured dots        each method's estimate, fading trail in the same colour, short tick = estimated heading
  grey wedge + arrow   the camera's true view direction and field of view
  green lines          true camera position -> every marker detected in this frame (marker turns green)
  error panel          position error vs. time (log scale), drawn up to "now", with a cursor
  red bands            periods with no marker visible (>= `min_gap_s` without any detection)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# Same hues as app/dashboard.py and vislam/viz.py so the replay matches every other figure.
COLORS = {"IMU only": "#d62728", "VIO": "#ff7f0e", "VIO + anchors (EKF)": "#2ca02c",
          "VIO + landmarks (graph SLAM)": "#1f77b4", "Relocalisation (map + EKF)": "#9467bd"}
_FALLBACK = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
NO_MARKER = "#d62728"
SEEN = "#2ca02c"


def color_of(name: str, i: int = 0) -> str:
    return COLORS.get(name, _FALLBACK[i % len(_FALLBACK)])


# ====================================================================================== data
@dataclass
class Replay:
    t: np.ndarray                         # (N,) camera-frame times [s]
    gt_p: np.ndarray | None               # (N,3) true position, None for real data without ground truth
    gt_dir: np.ndarray | None             # (N,3) true camera optical axis in world frame (unit)
    methods: dict                         # name -> dict(p=(N,3) NaN where invalid, dir=(N,3), err=(N,) or None)
    seen: list                            # per frame: list of marker ids detected in that frame
    marker_pos: dict                      # id -> world position (surveyed map, else estimated map)
    spans: list                           # [(t0, t1)] intervals with no marker visible
    fov_deg: float = 70.0
    cam_rate: float = 20.0
    bounds: tuple = (0, 6, 0, 5)          # xmin, xmax, ymin, ymax of the map view
    room: tuple | None = None             # (Lx, Ly) outline for simulated data
    name: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.t)

    def in_gap(self, k):
        """Interval index if frame k lies in a highlighted no-marker span, else None."""
        for j, (a, b) in enumerate(self.spans):
            if a <= self.t[k] <= b:
                return j
        return None


def no_marker_spans(t, seen, min_gap_s=1.0):
    """Maximal runs of frames without any detection that last at least `min_gap_s` seconds."""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    out, start = [], None
    for k in range(len(t)):
        empty = len(seen[k]) == 0
        if empty and start is None:
            start = k
        if start is not None and (not empty or k == len(t) - 1):
            end = k - 1 if not empty else k
            a, b = float(t[start]), float(t[end] + dt)
            if b - a >= min_gap_s:
                out.append((a, b))
            start = None
    return out


def build_replay(ds, trajs: dict, est_map=None, min_gap_s=1.0, name="") -> Replay:
    """Collect everything the renderers need from a Dataset and {method name: Trajectory}."""
    t = np.asarray(ds.cam_t, float)
    R_BC = ds.T_BC[:3, :3]
    axis = R_BC[:, 2]                                   # camera optical axis (z_C) expressed in the body frame

    def view_dir(R):
        d = np.einsum("nij,j->ni", R, axis)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    has_gt = ds.gt_p is not None
    methods = {}
    for nm, tr in trajs.items():
        p = np.where(tr.ok[:, None], tr.p, np.nan)
        err = np.where(tr.ok, np.linalg.norm(tr.p - ds.gt_p, axis=1), np.nan) if has_gt else None
        methods[nm] = dict(p=p, dir=view_dir(tr.R), err=err)

    by_frame = ds.obs_by_frame()
    seen = [sorted({i for i, _ in by_frame.get(k, [])}) for k in range(len(t))]
    mp = ds.map_gt if ds.map_gt else (est_map or {})
    marker_pos = {int(i): np.asarray(T)[:3, 3] for i, T in mp.items()}

    cfg = ds.meta.get("config", {})
    fov = float(cfg.get("fov_deg", ds.meta.get("fov_deg", 70.0)))
    cam_rate = float(ds.meta.get("cam_rate", 1.0 / np.median(np.diff(t)) if len(t) > 1 else 20.0))

    pts = [ds.gt_p[:, :2]] if has_gt else [m["p"][:, :2] for m in methods.values()]
    pts += [np.array([v[:2] for v in marker_pos.values()])] if marker_pos else []
    P = np.vstack([p[np.isfinite(p).all(1)] for p in pts if len(p)])
    if not has_gt:                                      # robust to a diverged method when there is no reference
        lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)
    else:
        lo, hi = P.min(0), P.max(0)
    pad = max(0.8, 0.08 * float((hi - lo).max()))
    bounds = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)
    room = None
    if ds.meta.get("simulated"):
        from .sim import ROOM
        room = ROOM[:2]
        bounds = (min(bounds[0], -0.5), max(bounds[1], room[0] + 0.5), min(bounds[2], -0.5), max(bounds[3], room[1] + 0.5))

    return Replay(t, ds.gt_p if has_gt else None, view_dir(ds.gt_R) if has_gt else None, methods, seen, marker_pos,
                  no_marker_spans(t, seen, min_gap_s), fov, cam_rate, bounds, room,
                  name or ds.meta.get("name", ""))


def _wedge(c, d, fov_deg, radius=1.2, n=9):
    """Top-down polygon of the camera's horizontal field of view (apex c, optical axis d)."""
    h = np.arctan2(d[1], d[0])
    ang = h + np.radians(np.linspace(-fov_deg / 2, fov_deg / 2, n))
    arc = np.column_stack([c[0] + radius * np.cos(ang), c[1] + radius * np.sin(ang)])
    return np.vstack([c[:2], arc, c[:2]])


def _tag(rp, k):
    """(text, colour) describing the marker status at frame k."""
    if rp.seen[k]:
        return "MARKER " + ", ".join(str(i) for i in rp.seen[k]) + " VISIBLE", SEEN
    j = rp.in_gap(k)
    if j is not None:
        a, b = rp.spans[j]
        return f"NO MARKER VISIBLE  ({rp.t[k] - a:.1f} s / {b - a:.1f} s)", NO_MARKER
    return "no detection in this frame", "#888888"


def _inside(rp, p):
    x0, x1, y0, y1 = rp.bounds
    return np.isfinite(p).all() and x0 <= p[0] <= x1 and y0 <= p[1] <= y1


# ====================================================================================== Plotly
def plotly_replay(rp: Replay, max_frames=240, trail_s=4.0, speed=1.0, height=720):
    """Interactive replay: Play/Pause buttons and a time slider.  Returns a plotly Figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    N = rp.n
    stride = max(1, int(np.ceil(N / max_frames)))
    ks = np.arange(0, N, stride)
    if ks[-1] != N - 1:
        ks = np.r_[ks, N - 1]
    tt = rp.t[ks]
    K = int(np.clip(trail_s * rp.cam_rate / stride, 8, 60))                    # trail points
    trail_step = max(1, int(round(trail_s * rp.cam_rate / K)))
    op = np.linspace(0.04, 0.75, K)                                           # fading trail
    names = list(rp.methods)
    cols = {n: color_of(n, i) for i, n in enumerate(names)}
    has_gt = rp.gt_p is not None

    fig = make_subplots(rows=1, cols=2, column_widths=[0.56, 0.44], horizontal_spacing=0.09)
    dyn = []                                   # (trace index, function k -> dict of data updates)

    def add(trace, row, col, fn=None):
        fig.add_trace(trace, row=row, col=col)
        if fn is not None:
            dyn.append((len(fig.data) - 1, fn))

    def trail_xy(P, k):
        idx = k - trail_step * np.arange(K)[::-1]
        pts = np.full((K, 2), np.nan)
        ok = idx >= 0
        pts[ok] = P[idx[ok], :2]
        return pts[:, 0], pts[:, 1]

    def blank(**kw):
        return dict(x=[None], y=[None], **kw)

    # ----------------------------- static scene
    if rp.room:
        Lx, Ly = rp.room
        add(go.Scatter(x=[0, Lx, Lx, 0, 0], y=[0, 0, Ly, Ly, 0], mode="lines", line=dict(color="#999", width=2),
                       hoverinfo="skip", showlegend=False), 1, 1)
    if has_gt:
        add(go.Scatter(x=rp.gt_p[:, 0], y=rp.gt_p[:, 1], mode="lines", line=dict(color="rgba(0,0,0,.18)", width=1, dash="dash"),
                       hoverinfo="skip", showlegend=False), 1, 1)
    if rp.marker_pos:
        ids = list(rp.marker_pos)
        add(go.Scatter(x=[rp.marker_pos[i][0] for i in ids], y=[rp.marker_pos[i][1] for i in ids], mode="markers+text",
                       text=[str(i) for i in ids], textposition="top right", textfont=dict(size=11, color="#555"),
                       marker=dict(symbol="square", size=9, color="#333"), name="Marker", hoverinfo="skip"), 1, 1)
        mpos = rp.marker_pos

        def det_markers(k):
            s = [i for i in rp.seen[k] if i in mpos]
            return dict(x=[mpos[i][0] for i in s] or [None], y=[mpos[i][1] for i in s] or [None])

        add(go.Scatter(**blank(), mode="markers", marker=dict(symbol="square", size=15, color="rgba(44,160,44,.35)",
                       line=dict(color=SEEN, width=2)), name="Marker detected", hoverinfo="skip"), 1, 1, det_markers)

        def det_lines(k):
            xs, ys = [], []
            c = (rp.gt_p[k] if has_gt else next(iter(rp.methods.values()))["p"][k])
            if np.isfinite(c).all():
                for i in rp.seen[k]:
                    if i in mpos:
                        xs += [c[0], mpos[i][0], None]; ys += [c[1], mpos[i][1], None]
            return dict(x=xs or [None], y=ys or [None])

        add(go.Scatter(**blank(), mode="lines", line=dict(color=SEEN, width=2, dash="dot"),
                       name="Line to detected marker", hoverinfo="skip"), 1, 1, det_lines)

    # ----------------------------- camera view direction (true)
    if has_gt:
        def fov(k):
            w = _wedge(rp.gt_p[k], rp.gt_dir[k], rp.fov_deg)
            return dict(x=w[:, 0], y=w[:, 1])

        add(go.Scatter(**blank(), mode="lines", fill="toself", fillcolor="rgba(0,0,0,.08)", line=dict(width=0),
                       name="Camera field of view", hoverinfo="skip"), 1, 1, fov)

        def axis_line(k):
            d = rp.gt_dir[k, :2] / max(np.linalg.norm(rp.gt_dir[k, :2]), 1e-9)
            c = rp.gt_p[k]
            return dict(x=[c[0], c[0] + 0.9 * d[0]], y=[c[1], c[1] + 0.9 * d[1]])

        add(go.Scatter(**blank(), mode="lines+markers", line=dict(color="black", width=2),
                       marker=dict(symbol="diamond", size=[0, 8], color="black"), name="Camera view direction",
                       hoverinfo="skip"), 1, 1, axis_line)

        add(go.Scatter(**blank(), mode="markers", marker=dict(size=5, color="black", opacity=op), showlegend=False,
                       hoverinfo="skip"), 1, 1, lambda k: dict(zip("xy", trail_xy(rp.gt_p, k))))
        # big black dot first => it sits under the (smaller) method dots and shows as a halo when they coincide
        add(go.Scatter(**blank(), mode="markers", marker=dict(size=19, color="black"), name="True position",
                       hoverinfo="skip"), 1, 1, lambda k: dict(x=[rp.gt_p[k, 0]], y=[rp.gt_p[k, 1]]))

    # ----------------------------- methods on the map
    for n in names:
        m = rp.methods[n]
        add(go.Scatter(**blank(), mode="markers", marker=dict(size=5, color=cols[n], opacity=op), legendgroup=n,
                       showlegend=False, hoverinfo="skip"), 1, 1, lambda k, m=m: dict(zip("xy", trail_xy(m["p"], k))))

        def tick(k, m=m):
            p, d = m["p"][k], m["dir"][k, :2]
            if not _inside(rp, p):
                return dict(x=[None], y=[None])
            d = d / max(np.linalg.norm(d), 1e-9)
            return dict(x=[p[0], p[0] + 0.35 * d[0]], y=[p[1], p[1] + 0.35 * d[1]])

        add(go.Scatter(**blank(), mode="lines", line=dict(color=cols[n], width=3), legendgroup=n, showlegend=False,
                       hoverinfo="skip"), 1, 1, tick)
        add(go.Scatter(**blank(), mode="markers", marker=dict(size=11, color=cols[n], line=dict(color="white", width=1.5)),
                       name=n, legendgroup=n, hoverinfo="skip"), 1, 1,
            lambda k, m=m: dict(x=[m["p"][k, 0]], y=[m["p"][k, 1]]))
    # ----------------------------- error-vs-time panel
    err_floor = 1e-3
    if has_gt:
        allv = np.concatenate([np.r_[m["err"][np.isfinite(m["err"])]] for m in rp.methods.values()])
        lo = max(err_floor, float(allv.min())) if len(allv) else err_floor
        hi = float(allv.max()) if len(allv) else 1.0
        yr = [np.log10(lo) - 0.15, np.log10(hi) + 0.15]
        for n in names:
            e = rp.methods[n]["err"][ks]
            add(go.Scatter(**blank(), mode="lines", line=dict(color=cols[n], width=2.5), legendgroup=n, showlegend=False,
                           hoverinfo="skip"), 1, 2,
                lambda k, e=e: dict(x=tt[: np.searchsorted(ks, k, side="right")].tolist(),
                                    y=[None if not np.isfinite(v) else float(v) for v in e[: np.searchsorted(ks, k, side="right")]]))
            add(go.Scatter(**blank(), mode="markers", marker=dict(size=9, color=cols[n], line=dict(color="white", width=1)),
                           legendgroup=n, showlegend=False, hoverinfo="skip"), 1, 2,
                lambda k, n=n: dict(x=[rp.t[k]], y=[None if not np.isfinite(rp.methods[n]["err"][k]) else float(rp.methods[n]["err"][k])]))
        add(go.Scatter(x=[rp.t[0]] * 2, y=[10 ** yr[0], 10 ** yr[1]], mode="lines", line=dict(color="#333", width=1.5),
                       showlegend=False, hoverinfo="skip"), 1, 2, lambda k: dict(x=[rp.t[k]] * 2))
        fig.update_yaxes(type="log", range=yr, title_text="position error [m]", row=1, col=2, gridcolor="rgba(128,128,128,.25)")
    else:
        fig.update_yaxes(visible=False, row=1, col=2)
    fig.update_xaxes(range=[rp.t[0], rp.t[-1]], title_text="time [s]", row=1, col=2, gridcolor="rgba(128,128,128,.25)")

    # no-marker-visible bands (static, behind the curves)
    shapes = []
    for a, b in rp.spans:
        shapes.append(dict(type="rect", xref="x2", yref="y2 domain", x0=a, x1=b, y0=0, y1=1, fillcolor="rgba(214,39,40,.16)",
                           line=dict(width=0), layer="below",
                           label=dict(text="no marker visible", textposition="top center", font=dict(size=10, color=NO_MARKER))))

    # ----------------------------- readout (updated every frame) + layout
    def readout(k):
        txt, colr = _tag(rp, k)
        lines = [f"<b>t = {rp.t[k]:5.1f} s</b>   <span style='color:{colr}'><b>{txt}</b></span>"]
        for n in names:
            m = rp.methods[n]
            if m["err"] is not None and np.isfinite(m["err"][k]):
                tail = "" if _inside(rp, m["p"][k]) else "  (off map)"
                lines.append(f"<span style='color:{cols[n]}'>●</span> {n}: <b>{m['err'][k]:.3f} m</b>{tail}")
            else:
                lines.append(f"<span style='color:{cols[n]}'>●</span> {n}: –")
        return [dict(text="<br>".join(lines), xref="paper", yref="paper", x=0.0, y=1.0, xanchor="left", yanchor="bottom",
                     showarrow=False, align="left", font=dict(size=12))]

    x0, x1, y0, y1 = rp.bounds
    fig.update_xaxes(range=[x0, x1], title_text="x [m]", row=1, col=1, gridcolor="rgba(128,128,128,.25)", constrain="domain")
    fig.update_yaxes(range=[y0, y1], title_text="y [m]", row=1, col=1, scaleanchor="x", scaleratio=1,
                     gridcolor="rgba(128,128,128,.25)")

    def frame_data(k):
        return [go.Scatter(**f(k)) for _, f in dyn]

    idx = [i for i, _ in dyn]
    for i, f in dyn:                                             # initial state = first frame
        fig.data[i].update(**f(0))
    dur = stride / rp.cam_rate / speed * 1000
    fig.frames = [go.Frame(data=frame_data(k), traces=idx, name=str(j), layout=dict(annotations=readout(k)))
                  for j, k in enumerate(ks)]
    fivesec = np.floor(tt / 5)
    steps = [dict(method="animate", label=(f"{tt[j]:.0f}s" if j == 0 or fivesec[j] != fivesec[j - 1] else " "),
                  args=[[str(j)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for j in range(len(ks))]
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=50, r=20, t=110, b=170), annotations=readout(0), shapes=shapes,
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0), uirevision="replay",
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.46, xanchor="left", yanchor="top", showactive=False, pad=dict(t=0, r=10),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(mode="immediate", fromcurrent=True, frame=dict(duration=dur, redraw=True),
                                                         transition=dict(duration=0))]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                                           transition=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.14, y=-0.46, len=0.86, xanchor="left", yanchor="top", pad=dict(t=0, b=0), steps=steps,
                      currentvalue=dict(visible=False), ticklen=3, minorticklen=0)],
    )
    return fig


# ====================================================================================== video
def render_video(rp: Replay, path="replay.mp4", fps=20, speed=1.0, trail_s=4.0, size=(1280, 720), progress=None,
                 max_seconds=None):
    """Render the replay to MP4 (H.264, plays in browsers/PowerPoint).  Falls back to GIF when no ffmpeg is available.

    fps    output frame rate           speed  playback speed relative to real time (2.0 = twice as fast)
    Returns the path actually written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Polygon

    N = rp.n
    step = max(1, int(round(speed * rp.cam_rate / fps)))
    frames = np.arange(0, N, step)
    if max_seconds:
        frames = frames[rp.t[frames] - rp.t[0] <= max_seconds]
    K = max(2, int(trail_s * rp.cam_rate))
    names = list(rp.methods)
    cols = {n: color_of(n, i) for i, n in enumerate(names)}
    has_gt = rp.gt_p is not None
    W, H = size
    dpi = 100

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1.5, 1], left=0.05, right=0.98, top=0.93, bottom=0.14,
                          wspace=0.16, hspace=0.18)
    axm, axe, axt = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    axt.axis("off")
    fig.suptitle(f"Replay — {rp.name}" if rp.name else "Replay", x=0.05, ha="left", fontsize=14, fontweight="bold")

    # ---- map
    x0, x1, y0, y1 = rp.bounds
    axm.set_xlim(x0, x1); axm.set_ylim(y0, y1); axm.set_aspect("equal"); axm.grid(alpha=.3)
    axm.set_xlabel("x [m]"); axm.set_ylabel("y [m]")
    if rp.room:
        Lx, Ly = rp.room
        axm.plot([0, Lx, Lx, 0, 0], [0, 0, Ly, Ly, 0], color="0.55", lw=2, zorder=1)
    if has_gt:
        axm.plot(rp.gt_p[:, 0], rp.gt_p[:, 1], "--", color="0.8", lw=1, zorder=1)
    mart = {}
    for i, p in rp.marker_pos.items():
        mart[i] = axm.plot(p[0], p[1], "s", ms=8, color="0.2", zorder=3)[0]
        axm.annotate(str(i), p[:2] + 0.1, fontsize=9, color="0.3", zorder=3)
    lines = [axm.plot([], [], ":", color=SEEN, lw=2, zorder=4)[0] for _ in range(4)]
    wedge = Polygon(np.zeros((3, 2)), closed=True, fc=(0, 0, 0, .08), ec=(0, 0, 0, .25), lw=.8, zorder=2)
    axm.add_patch(wedge)
    axis_ln, = axm.plot([], [], "-", color="k", lw=2, zorder=6)
    trails = {}
    for n in ["__gt"] + names:
        lc = LineCollection([], linewidths=2.2, zorder=5 if n != "__gt" else 4)
        axm.add_collection(lc)
        trails[n] = lc
    gt_dot, = axm.plot([], [], "o", ms=14, color="k", zorder=7)
    dots, ticks = {}, {}
    for n in names:
        ticks[n], = axm.plot([], [], "-", color=cols[n], lw=2.5, zorder=8)
        dots[n], = axm.plot([], [], "o", ms=8, color=cols[n], mec="white", mew=1.2, zorder=9, label=n)
    handles = ([gt_dot] if has_gt else []) + [dots[n] for n in names]
    if has_gt:
        gt_dot.set_label("True position")
    axm.legend(handles=handles, loc="upper left", fontsize=8, framealpha=.9, ncol=2, bbox_to_anchor=(0, -0.11), borderaxespad=0)

    # ---- error panel
    axe.set_xlim(rp.t[0], rp.t[-1]); axe.grid(alpha=.3, which="both"); axe.set_xlabel("time [s]")
    axe.set_title("Position error vs. time", fontsize=10, loc="left")
    curves, edots = {}, {}
    if has_gt:
        allv = np.concatenate([m["err"][np.isfinite(m["err"])] for m in rp.methods.values()])
        axe.set_yscale("log"); axe.set_ylabel("error [m]")
        axe.set_ylim(max(1e-3, allv.min()) / 1.4, allv.max() * 1.4)
        for n in names:
            axe.plot(rp.t, rp.methods[n]["err"], color=cols[n], lw=1, alpha=.15)
            curves[n], = axe.plot([], [], color=cols[n], lw=2.2)
            edots[n], = axe.plot([], [], "o", color=cols[n], ms=6, mec="white")
    else:
        axe.text(.5, .5, "no ground truth\n(error unavailable)", ha="center", va="center", transform=axe.transAxes, color="0.5")
    for j, (a, b) in enumerate(rp.spans):
        axe.axvspan(a, b, color=NO_MARKER, alpha=.14, lw=0, label="no marker visible" if j == 0 else None)
    cursor = axe.axvline(rp.t[0], color="0.2", lw=1.3)
    if rp.spans:
        axe.legend(loc="upper left", fontsize=8, framealpha=.9)

    # ---- readout
    time_txt = axt.text(0.0, 1.0, "", fontsize=15, fontweight="bold", va="top", transform=axt.transAxes)
    badge = axt.text(0.0, 0.83, "", fontsize=12.5, fontweight="bold", va="top", transform=axt.transAxes,
                     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7"))
    rows = {}
    for r, n in enumerate(names):
        y = 0.62 - 0.12 * r
        axt.plot([0.01], [y - 0.03], "o", ms=9, color=cols[n], transform=axt.transAxes, clip_on=False)
        rows[n] = axt.text(0.045, y, "", fontsize=11, va="top", transform=axt.transAxes)

    def draw_frame(k):
        k0 = max(0, k - K + 1)
        sl = slice(k0, k + 1)
        txt, colr = _tag(rp, k)
        gap = rp.in_gap(k) is not None and not rp.seen[k]
        axm.set_facecolor((1, .92, .92) if gap else "white")
        time_txt.set_text(f"t = {rp.t[k]:5.1f} s")
        badge.set_text(txt); badge.set_color(colr); badge.get_bbox_patch().set_edgecolor(colr)
        for i, a in mart.items():
            hit = i in rp.seen[k]
            a.set_color(SEEN if hit else "0.2"); a.set_markersize(13 if hit else 8)
        c = rp.gt_p[k] if has_gt else None
        if has_gt:
            gt_dot.set_data([c[0]], [c[1]])
            wedge.set_xy(_wedge(c, rp.gt_dir[k], rp.fov_deg))
            d = rp.gt_dir[k, :2] / max(np.linalg.norm(rp.gt_dir[k, :2]), 1e-9)
            axis_ln.set_data([c[0], c[0] + .9 * d[0]], [c[1], c[1] + .9 * d[1]])
            _fade(trails["__gt"], rp.gt_p[sl, :2], (0, 0, 0))
        ref = c if has_gt else rp.methods[names[0]]["p"][k]          # without ground truth: from the first method's estimate
        vis = [i for i in rp.seen[k] if i in rp.marker_pos] if np.isfinite(ref).all() else []
        for j, ln in enumerate(lines):
            if j < len(vis):
                m = rp.marker_pos[vis[j]]
                ln.set_data([ref[0], m[0]], [ref[1], m[1]])
            else:
                ln.set_data([], [])
        for n in names:
            m = rp.methods[n]
            p = m["p"][k]
            ok = np.isfinite(p).all()
            dots[n].set_data([p[0]] if ok else [], [p[1]] if ok else [])
            _fade(trails[n], m["p"][sl, :2], cols[n])
            if ok and _inside(rp, p):
                d = m["dir"][k, :2] / max(np.linalg.norm(m["dir"][k, :2]), 1e-9)
                ticks[n].set_data([p[0], p[0] + .35 * d[0]], [p[1], p[1] + .35 * d[1]])
            else:
                ticks[n].set_data([], [])
            e = m["err"]
            if e is not None:
                curves[n].set_data(rp.t[: k + 1], e[: k + 1])
                edots[n].set_data([rp.t[k]] if np.isfinite(e[k]) else [], [e[k]] if np.isfinite(e[k]) else [])
                rows[n].set_text(f"{n}: {e[k]:.3f} m" + ("" if _inside(rp, p) else "  (off map)") if np.isfinite(e[k]) else f"{n}: –")
            else:
                rows[n].set_text(f"{n}" + ("" if ok else ": –"))
        cursor.set_xdata([rp.t[k]] * 2)

    def _fade(lc, pts, rgb):
        ok = np.isfinite(pts).all(1)
        pts = pts[ok]
        if len(pts) < 2:
            lc.set_segments([]); return
        seg = np.stack([pts[:-1], pts[1:]], 1)
        a = np.linspace(0.03, 0.8, len(seg))
        lc.set_segments(seg)
        lc.set_colors([(*to_rgba(rgb)[:3], x) for x in a])

    writer, out = _open_writer(path, fps, size)
    try:
        for n_done, k in enumerate(frames):
            draw_frame(int(k))
            fig.canvas.draw()
            writer(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            if progress:
                progress((n_done + 1) / len(frames))
    finally:
        writer(None)
        plt.close(fig)
    return out


def _open_writer(path, fps, size):
    """Return (write(frame_or_None), actual_path).  H.264 MP4 through imageio-ffmpeg, else animated GIF."""
    W, H = size
    if path.lower().endswith(".mp4"):
        try:
            import imageio_ffmpeg
            gen = imageio_ffmpeg.write_frames(path, (W, H), fps=fps, codec="libx264", pix_fmt_in="rgb24", macro_block_size=1,
                                              pix_fmt_out="yuv420p", output_params=["-preset", "veryfast", "-crf", "20", "-movflags", "+faststart"],
                                              quality=None)
            gen.send(None)

            def write(frame):
                if frame is None:
                    gen.close()
                else:
                    gen.send(np.ascontiguousarray(frame))
            return write, path
        except Exception as e:                                   # no ffmpeg available -> GIF
            print(f"[replay] MP4 unavailable ({e}); writing GIF instead")
            path = os.path.splitext(path)[0] + ".gif"
    from PIL import Image
    buf = []

    def write(frame):
        if frame is None:
            if buf:
                buf[0].save(path, save_all=True, append_images=buf[1:], duration=int(1000 / fps), loop=0, optimize=False)
        else:
            buf.append(Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE))
    return write, path
