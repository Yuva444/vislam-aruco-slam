# System design

## 1. Coordinate frames (`T_AB` maps B-coordinates into A)
| Frame | Definition |
|---|---|
| W | Room frame, z up, gravity **g** = (0,0,−9.81). Defined by the start pose of the mapping run (or surveyed markers). |
| B | IMU/body, x forward, y left, z up. |
| C | Camera (OpenCV): x right, y down, z forward. `T_BC` is the extrinsic (default: pure rotation, lever arm 0). |
| M_i | Marker i: x right, y up, z out of the face. `T_WM_i` stored in the map JSON. |

Camera pose from a sighting: `T_WC = T_WM_i · (T_CM_i)^-1`, body pose `T_WB = T_WC · T_BC^-1`.
A new marker is initialised with `T_WM_i = T_WC · T_CM_i` (landmark role).

## 2. Estimator 1 — error-state EKF (`vislam/ekf.py`)
Nominal state `p, v ∈ W`, `R = R_WB`, `b_a, b_g`; error state `δx = [δp, δv, δθ, δb_a, δb_g]` (15), `R_true = R·Exp(δθ)`.

IMU: `f_m = Rᵀ(a − g) + b_a + n_a`, `ω_m = ω + b_g + n_g`.  
Propagation: `p⁺ = p + v dt + ½ a dt²`, `v⁺ = v + a dt`, `R⁺ = R Exp((ω_m − b_g)dt)`, `a = R(f_m − b_a) + g`.  
Jacobian: `F = [[I, I dt, 0, 0, 0], [0, I, −R[f_m−b_a]× dt, −R dt, 0], [0, 0, Exp(−ω dt)ᵀ, 0, −I dt], [0,0,0,I,0], [0,0,0,0,I]]`.

Updates (Joseph form, sequential):
1. **Marker pose**: measurement `T_WB^meas = T_WM T_CM⁻¹ T_BC⁻¹`; residual `r = [p^meas − p ; Log(Rᵀ R^meas)]`, `H = [I 0 0 0 0; 0 0 I 0 0]`; noise grows with distance `σ_p = a + b·d`, `σ_θ = a' + b'·d`. Mahalanobis gate `rᵀS⁻¹r < χ²₆(99.9 %) = 22.46` rejects false detections.
2. **VO velocity** `z = Rᵀ v`: `H_v = Rᵀ`, `H_θ = [Rᵀ v]×`.
3. **VO direction** (monocular, scale-free) `u = Rᵀ v/‖Rᵀ v‖`: Jacobian through the normalisation `(I − u uᵀ)/‖w‖`.

Relocalisation: when the pose is unknown, the first map-marker sighting **resets** `p, R` (`v = 0`, large `σ_v`); filtering a large initial error through a linearised update diverges/gets gated.

## 3. Estimator 2 — landmark graph SLAM (`vislam/graph.py`)
Variables: keyframe poses `X_k` (every 0.5 s), marker poses `L_i` (6 DoF each).  
```
min Σ‖X_0 ⊖ T_0‖²_prior + Σ‖(ΔT_k^VIO)⁻¹ (X_k⁻¹X_{k+1})‖²_Σodo + Σ ρ(‖ z_ki ⊖ (X_k T_off T_BC)⁻¹ L_i ‖²_Λki)
```
`ΔT^VIO` comes from the EKF (VIO) front-end; marker observations are attached to the nearest keyframe through the
VIO offset `T_off`. Solved with sparse trust-region least squares + Huber loss (`jac_sparsity` from the factor structure).
Odometry covariance is deliberately inflated (`σ_t = 5 mm + 15 %·‖Δt‖`) because scale/bias errors are *correlated* along the path.
**Loop closure** is implicit: re-observing marker i ties together keyframes that are far apart in time.

## 4. Observability: how markers change drift
* Visual–inertial odometry has 4 unobservable directions: global position (3) and yaw. Roll/pitch/scale/velocity/biases are observable with excitation.
* One marker pose measurement observes the full 6-DoF body pose ⇒ removes position **and yaw** drift (a *known-map* marker; unknown landmarks only add relative constraints).
* Hence anchors (known `T_WM`) bound the error; landmarks only reduce it where they are re-observed (loop closure).

## 5. Error sources
Camera calibration error · rolling shutter · motion blur · PnP ambiguity at oblique angles / small markers ·
marker-size measurement error · camera–IMU time offset (shake test estimates it) · extrinsic error ·
accelerometer/gyro bias drift · VO scale drift · map error (surveying) · false detections.

## 6. Data flow & timing
IMU 200 Hz drives propagation; each camera frame (~20–30 Hz) triggers the marker/VO updates with a split-propagation to the exact frame time.
Camera time = IMU time + `cam_imu_offset_s`, estimated with `calibration.estimate_time_offset`.
EKF runs ≈ 50× real time on a laptop (26 s dataset in ≈ 0.5 s); graph SLAM ≈ 5–10 s offline.

## 7. Calibration procedure
Intrinsics: 15–25 checkerboard views covering corners/tilts, `calibrate_camera.py`, report RMS & per-image error.
Extrinsic: mount rigidly; set `--R-BC` from the CAD/hand measurement (lever arm neglected: state the assumption).
Time offset: 3 s vigorous rotation shake at the start; cross-correlate |ω_gyro| with |ω_visual| (marker PnP rotation rate).
