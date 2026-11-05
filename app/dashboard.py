"""VISLAM-ArUco dashboard.   Run:  streamlit run app/dashboard.py

Design a walking path (or press 🎲 for a random one) -> simulate -> every method is run -> results, replay and analysis."""
import glob, json, os, sys, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vislam.dataset import Dataset, save_map, load_map
from vislam.diagnostics import sensor_health
from vislam.metrics import map_metrics, detection_stats
from vislam.paths import PRESETS, ROOM, path_curve, path_length, random_path, validate_path
from vislam.pipeline import METHODS, run_method
from vislam.replay import COLORS, build_replay, plotly_replay, render_video
from vislam.report import gap_stats, method_table, narrative
from vislam.sim import SimConfig, SLOTS, simulate

st.set_page_config(page_title="VISLAM-ArUco", page_icon="📍", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
h1 {font-weight: 750; letter-spacing: -0.5px; font-size: 2.2rem !important;}
div[data-testid="stMetric"] {background:#f3f6fb; border:1px solid #e3e9f3; border-radius:12px; padding:12px 16px;}
.stButton > button[kind="primary"] {font-weight:650; padding:0.6rem 1.4rem;}
.hint {color:#6b7280; font-size:0.92rem;}
</style>""", unsafe_allow_html=True)
S = st.session_state

DEFAULT_METHODS = ["IMU only", "VIO", "VIO + anchors (EKF)", "VIO + landmarks (graph SLAM)"]
DIFFICULTY = {
    "Easy": dict(imu_noise_scale=0.5, det_noise_scale=0.5, p_miss=0.05),
    "Normal": dict(),
    "Hard": dict(imu_noise_scale=2.0, det_noise_scale=2.0, p_miss=0.4, p_outlier=0.02, sway_deg=60.0),
}
EXPLAIN = {"ate_rmse_m": "Average position error [m] (ATE RMSE) — lower is better",
           "ate_max_m": "Worst position error [m]", "rot_rmse_deg": "Average heading error [°]",
           "endpoint_drift_m": "Error at the end of the walk [m]", "drift_per_m": "End error divided by distance walked",
           "runtime_s": "Computation time [s]"}


# ====================================================================================== helpers
def run_analysis(ds, methods, map_file=None):
    amap = None
    if map_file and not map_file.startswith("surveyed"):
        amap = load_map(map_file)
    res = {"traj": {}, "est_map": None, "info": None}
    bar = st.progress(0.0, "Starting …")
    for n, m in enumerate(methods):
        bar.progress(n / len(methods), f"Running {m} …")
        tr, mk, info = run_method(ds, m, amap)
        res["traj"][m] = tr
        if mk is not None:
            res["est_map"], res["info"] = mk, info
    bar.empty()
    return res


def room_fig(wp=None, height=430):
    f = go.Figure()
    f.add_shape(type="rect", x0=0, y0=0, x1=ROOM[0], y1=ROOM[1], line=dict(color="#9ca3af", width=2), fillcolor="#f7f9fc", layer="below")
    P = np.array([(x, y) for x, y, *_ in SLOTS.values()])
    f.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers+text", text=[f"ID {i}" for i in SLOTS], textposition="top center",
                           marker=dict(symbol="square", size=10, color="#111827"), name="ArUco markers", hoverinfo="skip"))
    if wp is not None and len(wp) >= 2 and np.isfinite(wp).all():
        c = path_curve(wp)
        f.add_trace(go.Scatter(x=c[:, 0], y=c[:, 1], mode="lines", line=dict(color="#2563eb", width=3), name="Smoothed path", hoverinfo="skip"))
        f.add_trace(go.Scatter(x=wp[:, 0], y=wp[:, 1], mode="markers+text", text=[str(i + 1) for i in range(len(wp))],
                               textposition="middle center", textfont=dict(color="white", size=10),
                               marker=dict(size=19, color="#2563eb"), name="Waypoints"))
        f.add_trace(go.Scatter(x=[wp[0, 0]], y=[wp[0, 1]], mode="markers", marker=dict(size=27, color="rgba(34,197,94,.35)",
                               line=dict(color="#16a34a", width=2)), name="Start", hoverinfo="skip"))
    f.update_xaxes(range=[-0.5, ROOM[0] + 0.5], title="x [m]", gridcolor="#eef1f6", zeroline=False)
    f.update_yaxes(range=[-0.5, ROOM[1] + 0.5], title="y [m]", scaleanchor="x", gridcolor="#eef1f6", zeroline=False)
    f.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15))
    return f


def set_waypoints(wp):
    S.wp = [[float(x), float(y)] for x, y in wp]
    S.wp_ver = S.get("wp_ver", 0) + 1
    st.rerun()


# ====================================================================================== step 1: data
def planner():
    S.setdefault("wp", [list(p) for p in PRESETS["Rectangle loop"]])
    S.setdefault("wp_ver", 0)
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown("**① Choose how the person walks**")
        c = st.columns([2, 1])
        preset = c[0].selectbox("Preset path", list(PRESETS), label_visibility="collapsed")
        if c[1].button("Use preset", use_container_width=True):
            set_waypoints(PRESETS[preset])
        r = st.columns([1, 1])
        npts = r[0].number_input("Random path: waypoints", 3, 8, 5, help="How many corners the random path has")
        closed = r[1].checkbox("Loop back to start", False, help="Lets graph SLAM close a loop", key="closed_loop")
        if st.button("🎲 Generate a random path", use_container_width=True):
            seed = int(np.random.default_rng().integers(1_000_000))
            set_waypoints(random_path(seed, int(npts), closed))
        st.markdown('<span class="hint">…or edit / add / delete waypoints below (room is 6 × 5 m, keep 0.5 m from the walls). '
                    'Click the empty bottom row to add a point.</span>', unsafe_allow_html=True)
        df = pd.DataFrame(S.wp, columns=["x [m]", "y [m]"])
        ed = st.data_editor(df, num_rows="dynamic", hide_index=True, use_container_width=True, key=f"wp_editor_{S.wp_ver}",
                            column_config={"x [m]": st.column_config.NumberColumn(min_value=0.0, max_value=ROOM[0], step=0.1, format="%.2f"),
                                           "y [m]": st.column_config.NumberColumn(min_value=0.0, max_value=ROOM[1], step=0.1, format="%.2f")})
    wp = ed.to_numpy(float) if len(ed) else np.zeros((0, 2))
    errs, warns = validate_path(wp)
    with right:
        st.plotly_chart(room_fig(wp if not errs else None), use_container_width=True)
    for e in errs:
        st.error(e)
    for w in warns:
        st.warning(w)

    st.markdown("**② Set the conditions**")
    o = st.columns(4)
    speed = o[0].slider("Walking speed [m/s]", 0.4, 1.4, 0.8, 0.1)
    diff = o[1].select_slider("Sensor difficulty", list(DIFFICULTY), "Normal",
                              help="Easy = clean sensors, markers rarely missed. Hard = noisy IMU & detections, 40 % missed markers, false detections, fast camera sway.")
    seed = o[2].number_input("Noise seed", 0, 9999, 0, help="Same seed + same path = identical result")
    L = path_length(wp) if len(wp) > 1 else 0.0
    dur = L / speed + 1.5
    blk = o[3].checkbox("Hide all markers for a while", False, help="Simulates the camera being blocked / pointed at a blank wall")
    blackouts = ()
    if blk and dur > 8:
        b = st.slider("Marker blackout window [s]", 0.0, float(round(dur, 1)), (float(round(dur * 0.35, 1)), float(round(dur * 0.35 + min(12, dur * 0.4), 1))), 0.5)
        blackouts = ((b[0], b[1]),)
    st.caption(f"Path length {L:.1f} m · walk takes ≈ {dur:.0f} s · {len(wp)} waypoints")

    if st.button("▶ Simulate & analyse", type="primary", disabled=bool(errs)):
        with st.status("Simulating the walk …", expanded=False) as stt:
            cfg = SimConfig(path=[(float(x), float(y)) for x, y in wp], seed=int(seed), speed=float(speed), blackouts=blackouts,
                            name="My path", **DIFFICULTY[diff])
            S.ds = simulate(cfg)
            stt.update(label="Running IMU-only, VIO, anchored EKF and graph SLAM …")
        S.res = run_analysis(S.ds, DEFAULT_METHODS)
        S.run_info = dict(wp=wp.tolist(), speed=speed, diff=diff, seed=int(seed), blackouts=blackouts, at=time.strftime("%H:%M:%S"))
        S.pop("replay_video", None)
        st.rerun()


def dataset_picker():
    found = sorted(os.path.dirname(p) for p in glob.glob(f"{ROOT}/data/**/imu.csv", recursive=True))
    if not found:
        st.info("No datasets in the data/ folder yet. Use 'Design a path' or record/convert your own.")
        return
    c = st.columns([3, 1])
    pick = c[0].selectbox("Dataset folder", found, format_func=os.path.basename)
    if c[1].button("Load & analyse", type="primary", use_container_width=True):
        S.ds = Dataset.load(pick)
        S.res = run_analysis(S.ds, DEFAULT_METHODS)
        S.run_info = dict(at=time.strftime("%H:%M:%S"), source=os.path.basename(pick))
        S.pop("replay_video", None)
        st.rerun()


def webcam_recorder():
    secs = st.number_input("Duration [s]", 5, 600, 30)
    name = st.text_input("Name", "run1")
    if st.button("● Start recording"):
        from record_webcam import record
        with st.spinner("Recording …"):
            n, d = record(f"{ROOT}/data/raw/{name}", secs)
        st.success(f"{n} frames in {d:.1f}s → data/raw/{name}. Add the IMU CSV and run scripts/make_dataset.py.")


# ====================================================================================== step 2: results
def results():
    ds, res = S.ds, S.res
    info = S.get("run_info", {})
    rp = build_replay(ds, res["traj"], res["est_map"], 1.0, ds.meta.get("name", ""))
    table = method_table(ds, res["traj"])
    st.divider()
    st.subheader("Results")
    if info.get("at"):
        st.caption(f"Run at {info['at']}" + (f" · speed {info['speed']} m/s · difficulty {info['diff']}" if "speed" in info else ""))

    ate = {n: m.get("ate_rmse_m") for n, m in table.items() if m}
    if ate:
        cols = st.columns(len(ate))
        best = min(ate, key=ate.get)
        for c, (n, v) in zip(cols, ate.items()):
            c.metric(n, f"{v:.3f} m" if v < 10 else f"{v:.1f} m", "★ best" if n == best else None, delta_color="off")
        st.caption("Average position error against the true path (lower is better).")
    with st.container(border=True):
        st.markdown("##### What happened")
        for line in narrative(ds, rp, table):
            st.markdown(f"- {line}")

    t_rep, t_err, t_map, t_tab, t_data = st.tabs(["🎬 Replay", "📉 Error analysis", "🗺️ Map", "📋 Metrics", "🔍 Markers & sensors"])

    with t_rep:
        st.caption("Press ▶ Play or drag the slider. Black dot = true position · coloured dots = estimates · grey wedge = camera view direction · "
                   "green dotted lines = markers detected right now · red band = no marker visible.")
        c = st.columns([3, 1, 1, 1])
        pick = c[0].multiselect("Methods shown", list(res["traj"]), list(res["traj"]), key="rp_methods")
        speed = c[1].select_slider("Speed", [0.5, 1, 2, 4], 1, key="rp_speed", format_func=lambda v: f"{v}×")
        trail = c[2].slider("Trail [s]", 1.0, 10.0, 4.0, key="rp_trail")
        gap = c[3].slider("Min. gap [s]", 0.5, 10.0, 1.0, key="rp_gap", help="Marker-free stretches at least this long are shaded red")
        if not pick:
            st.info("Select at least one method.")
        else:
            rp2 = build_replay(ds, {n: res["traj"][n] for n in pick}, res["est_map"], gap, ds.meta.get("name", ""))
            if ds.gt_p is None:
                st.warning("No ground truth in this dataset: only estimates are animated and there is no error curve.")
            st.plotly_chart(plotly_replay(rp2, trail_s=trail, speed=speed), use_container_width=True)
            nfr = len(range(0, rp2.n, max(1, round(speed * rp2.cam_rate / 20))))
            if st.button(f"🎞 Export as MP4 video  (≈ {nfr * 0.12:.0f} s to render)"):
                bar = st.progress(0.0, "Rendering video …")
                os.makedirs(f"{ROOT}/results", exist_ok=True)
                S.replay_video = render_video(rp2, f"{ROOT}/results/replay_{ds.meta.get('name') or 'run'}.mp4".replace(" ", "_"), fps=20,
                                              speed=speed, trail_s=trail, progress=lambda x: bar.progress(x, "Rendering video …"))
                bar.empty()
            vid = S.get("replay_video")
            if vid and os.path.exists(vid):
                (st.video if vid.endswith(".mp4") else st.image)(vid)
                with open(vid, "rb") as fh:
                    st.download_button("⬇ Download " + os.path.basename(vid), fh.read(), os.path.basename(vid))

    with t_err:
        if ds.gt_p is None:
            st.info("Error analysis needs ground truth, which this dataset does not have.")
        else:
            f = go.Figure()
            for n, m in rp.methods.items():
                f.add_trace(go.Scatter(x=rp.t, y=m["err"], mode="lines", name=n, line=dict(color=COLORS.get(n), width=2.5)))
            for a, b in rp.spans:
                f.add_vrect(x0=a, x1=b, fillcolor="rgba(214,39,40,.12)", line_width=0, annotation_text="no marker visible",
                            annotation_position="top left", annotation_font=dict(color="#d62728", size=11))
            top = max(np.nanmax(m["err"]) for m in rp.methods.values())
            f.update_yaxes(type="log", range=[-3, float(np.ceil(np.log10(top * 1.2)))], title="position error [m] (log scale)", gridcolor="#eef1f6")
            f.update_xaxes(title="time [s]", gridcolor="#eef1f6")
            f.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(f, use_container_width=True)
            gs = gap_stats(rp)
            rows = {n: {"Average error with markers [m]": d["mean_with_markers"], "Average error in marker-free gaps [m]": d["mean_in_gap"],
                        "Worst error in a gap [m]": d["worst_in_gap"]} for n, d in gs.items()}
            st.markdown("**Does losing the markers hurt?**" if rp.spans else "**No marker-free gap longer than 1 s in this run.**")
            if rp.spans:
                st.dataframe(pd.DataFrame(rows).T.round(3), use_container_width=True)
            bar = go.Figure(go.Bar(x=list(ate), y=list(ate.values()), marker_color=[COLORS.get(n, "#888") for n in ate],
                                   text=[f"{v:.3f}" for v in ate.values()], textposition="outside"))
            bar.update_layout(template="plotly_white", height=330, yaxis_title="ATE RMSE [m]", yaxis_type="log", margin=dict(l=10, r=10, t=30, b=10),
                              title="Average error per method")
            st.plotly_chart(bar, use_container_width=True)

    with t_map:
        view = st.radio("View", ["Top-down", "3D"], horizontal=True)
        fig = go.Figure()
        if view == "3D":
            T3 = lambda **k: go.Scatter3d(**k)
        else:
            T3 = lambda **k: go.Scatter({q: v for q, v in k.items() if q != "z"})
        if ds.gt_p is not None:
            fig.add_trace(T3(x=ds.gt_p[:, 0], y=ds.gt_p[:, 1], z=ds.gt_p[:, 2], mode="lines", name="Ground truth", line=dict(color="black", dash="dash")))
        for n, tr in res["traj"].items():
            p = tr.p[tr.ok]
            fig.add_trace(T3(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines", name=n, line=dict(color=COLORS.get(n), width=4)))
        for lbl, mp, sym, col in [("Surveyed markers", ds.map_gt, "square", "black"), ("Estimated markers (SLAM)", res["est_map"], "x", "blue")]:
            if mp:
                P = np.array([T[:3, 3] for T in mp.values()])
                fig.add_trace(T3(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers+text", text=[str(i) for i in mp], name=lbl,
                                 marker=dict(symbol=sym, size=7 if view == "Top-down" else 5, color=col)))
        fig.update_layout(template="plotly_white", height=640, legend=dict(orientation="h", y=-0.1), margin=dict(l=10, r=10, t=10, b=10))
        if view == "3D":
            fig.update_layout(scene=dict(aspectmode="data"))
        else:
            fig.update_xaxes(title="x [m]", gridcolor="#eef1f6"); fig.update_yaxes(title="y [m]", scaleanchor="x", gridcolor="#eef1f6")
        # IMU-only can drift far outside the room; keep the view on the room
        if view == "Top-down" and ds.gt_p is not None:
            lo, hi = ds.gt_p[:, :2].min(0) - 1.2, ds.gt_p[:, :2].max(0) + 1.2
            fig.update_xaxes(range=[min(lo[0], -0.5), max(hi[0], ROOM[0] + 0.5)]); fig.update_yaxes(range=[min(lo[1], -0.5), max(hi[1], ROOM[1] + 0.5)])
        st.plotly_chart(fig, use_container_width=True)
        if res["est_map"] and st.button("💾 Save estimated marker map to results/"):
            os.makedirs(f"{ROOT}/results", exist_ok=True)
            save_map(f"{ROOT}/results/map_saved.json", res["est_map"], ds.meta.get("marker_size", 0))
            st.success("Saved results/map_saved.json — it can be used as an anchor / relocalisation map in later runs.")

    with t_tab:
        st.caption("Hover a column name for its meaning.")
        cols = ["ate_rmse_m", "ate_max_m", "rot_rmse_deg", "endpoint_drift_m", "drift_per_m", "runtime_s"]
        df = pd.DataFrame({n: {c: m.get(c) for c in cols} for n, m in table.items()}).T
        if res["est_map"] and ds.map_gt and "VIO + landmarks (graph SLAM)" in df.index:
            mm = map_metrics(res["est_map"], ds.map_gt)
            st.write(f"Graph-SLAM marker map: **{mm.get('n_markers_mapped', 0)}** markers mapped, mean position error "
                     f"**{mm.get('marker_pos_err_mean_m', float('nan')) * 100:.1f} cm**.")
        st.dataframe(df.round(3), use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(c, help=EXPLAIN[c]) for c in cols})
        if res["info"]:
            with st.expander("Graph optimisation details"):
                st.json(res["info"])

    with t_data:
        d = detection_stats(ds)
        c = st.columns(4)
        c[0].metric("Marker detections", d["n_detections"]); c[1].metric("Frames with a marker", f"{d['frames_with_marker_pct']:.0f} %")
        c[2].metric("Longest marker-free gap", f"{d['longest_marker_free_gap_s']:.1f} s")
        if "detection_success_pct" in d:
            c[3].metric("Detection success (sim)", f"{d['detection_success_pct']:.0f} %")
        st.markdown("**Detections per marker ID**")
        st.bar_chart(pd.Series(ds.obs_id).value_counts().sort_index(), height=200)
        fr = np.zeros(len(ds.cam_t)); fr[np.unique(ds.obs_frame)] = 1
        st.markdown("**Was any marker visible?** (1 = yes)")
        st.area_chart(pd.DataFrame({"marker visible": fr}, index=ds.cam_t), height=120)
        h = sensor_health(ds)
        c = st.columns(5)
        c[0].metric("IMU rate", f"{h['imu_rate_hz']:.0f} Hz", f"jitter {h['imu_jitter_ms']:.2f} ms")
        c[1].metric("Camera rate", f"{h['cam_rate_hz']:.1f} fps", f"{h['cam_dropped_frames']} dropped")
        c[2].metric("Duration", f"{h['duration_s']:.1f} s")
        c[3].metric("Cam–IMU offset", f"{h['cam_imu_time_offset_s'] * 1e3:.0f} ms")
        c[4].metric("|accel| mean", f"{h['accel_norm_mean']:.2f} m/s²", "≈9.81 ⇒ units OK")
        st.line_chart(pd.DataFrame(ds.imu_acc[::10], index=ds.imu_t[::10], columns=["ax", "ay", "az"]), height=170)
        st.line_chart(pd.DataFrame(ds.imu_gyr[::10], index=ds.imu_t[::10], columns=["gx", "gy", "gz"]), height=170)


# ====================================================================================== page
st.title("📍 ArUco visual–inertial positioning")
st.markdown('<span class="hint">Draw a walking path, press <b>Simulate &amp; analyse</b>, and see how well a camera + motion sensor can '
            'track the walker — with and without printed markers on the walls.</span>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Data source")
    src = st.radio("Source", ["Design a path (simulate)", "Load saved dataset", "Record webcam"], label_visibility="collapsed")
    st.caption("Use *Design a path* to try ideas without any hardware. The other two are for real or previously saved recordings.")
    if "res" in S and st.button("🧹 Clear results"):
        for k in ("ds", "res", "run_info", "replay_video"):
            S.pop(k, None)
        st.rerun()

has_res = S.get("res") is not None
with st.expander("1 · Data" if src != "Design a path (simulate)" else "1 · Design your path", expanded=not has_res):
    {"Design a path (simulate)": planner, "Load saved dataset": dataset_picker, "Record webcam": webcam_recorder}[src]()

if has_res:
    results()
else:
    st.info("Pick or draw a path above and press **▶ Simulate & analyse**. It takes about 10–20 seconds.")
