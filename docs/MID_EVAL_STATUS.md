# Mid-evaluation status (M1–M7)

| Milestone | Evidence in repo | Status |
|---|---|---|
| M1 formulation, architecture, diagram | `README.md`, `docs/SYSTEM_DESIGN.md` | done (Architecture C/E: phone→laptop offline + dashboard) |
| M2 acquisition, timestamps, calibration | `scripts/record_webcam.py`, `make_dataset.py`, `calibrate_camera.py`, `calibration.py` (offset estimator, tested) | code done — **TODO: run on your camera, report RMS px** |
| M3 ArUco detection + pose | `vislam/aruco.py`, `tests/test_aruco_pose.py` (synthetic accuracy < 2 cm, < 3°) | code done — **TODO: range/accuracy table on real markers** |
| M4 VIO baseline without markers | `vislam/ekf.py`, `run_filter(marker_map=None)` | done (sim) — TODO real data |
| M5 marker map + visualisation | `vislam/graph.py`, dashboard, `results/map_closed_loop.json` | done (sim) |
| M6 drift correction / loop closure / relocalisation + baseline comparison | `experiments/run_experiments.py`, `results/` | done (sim, 5 seeds) |
| M7 final experiments, report, video | `docs/DATA_COLLECTION.md` protocol | **TODO: real experiments, report, demo video** |
