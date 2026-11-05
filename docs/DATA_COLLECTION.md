# Experimental protocol (handout §8–9)

## Marker layout (fill in; commit a photo + floor plan)
| ID | x [m] | y [m] | z [m] | wall normal | size [mm] | how measured |
|---|---|---|---|---|---|---|
| 0 | | | | | 180 | tape / laser |

Use 6–10 markers, heights 1.0–1.8 m, on at least 3 walls, size measured with a ruler after printing.

## The five experiments (≥3 repeats each; keep raw data)
| # | Path | What to log | Metrics |
|---|---|---|---|
| E1 | Straight / L / rectangle with floor tape | video, IMU, tape waypoints | ATE, endpoint drift, drift/m |
| E2 | Walk through a region with no marker visible (≥ 5 s) | as above + blackout times | error growth during gap, recovery error |
| E3 | Return to start and re-observe earlier markers | as above | loop residual, SLAM vs VIO |
| E4 | New run in the same room using the saved map; start at a *different* spot | as above | reloc time, reloc error |
| E5 | Dim light / fast pan / oblique angle / partial occlusion (pick one, justify) | as above | detection %, failure rate, drift |

Compare each run: IMU only · VIO · VIO + anchors · VIO + landmarks (graph SLAM).

## Ground truth options (state which you used)
Floor tape with measured waypoints (mark the time you cross each one with a clap/shake or a visible marker),
laser-measured marker positions (map error), end-point return error (closed loop: should be 0).
Provide `gt.csv` (`t,x,y,z`) and `map_gt.json` (same format as `results/map_closed_loop.json`).

## Checklist before each recording
Fixed focus/exposure · calibration file matches the camera mode (resolution!) · IMU rate ≥ 100 Hz · 3 s shake at start ·
start standing still for 2 s (gravity alignment) · note lighting/obstacles in `meta.json`.
