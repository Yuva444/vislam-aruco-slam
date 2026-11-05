# Simulated experiment results (mean ± std over 5 seeds)

## E1_simple_path

Detection: {'frames_with_marker_pct': 94.47852760736197, 'longest_marker_free_gap_s': 0.10000000000000053, 'n_detections': 174, 'detection_success_pct': 92.55319148936171}

| Method | ATE RMSE [m] | Rot RMSE [deg] | End drift [m] | Drift/m | Marker err [m] |
|---|---|---|---|---|---|
| IMU only | 1.880 ± 0.870 | 1.496 ± 0.497 | 4.086 ± 1.875 | 0.732 ± 0.336 | – |
| VIO | 0.123 ± 0.068 | 1.062 ± 0.633 | 0.173 ± 0.092 | 0.031 ± 0.016 | – |
| VIO + anchors (EKF) | 0.041 ± 0.014 | 0.551 ± 0.122 | 0.025 ± 0.012 | 0.005 ± 0.002 | – |
| VIO + landmarks (graph SLAM) | 0.095 ± 0.032 | 1.768 ± 0.422 | 0.138 ± 0.053 | 0.025 ± 0.010 | 0.110 ± 0.031 |

## E2_marker_loss

Detection: {'frames_with_marker_pct': 48.490566037735846, 'longest_marker_free_gap_s': 12.049999999999999, 'n_detections': 295, 'detection_success_pct': 51.936619718309856}

| Method | ATE RMSE [m] | Rot RMSE [deg] | End drift [m] | Drift/m | Loop residual [m] | Marker err [m] |
|---|---|---|---|---|---|---|
| IMU only | 25.011 ± 12.532 | 2.205 ± 1.447 | 54.878 ± 27.843 | 2.502 ± 1.269 | 54.882 ± 27.842 | – |
| VIO | 0.135 ± 0.019 | 2.656 ± 0.919 | 0.216 ± 0.049 | 0.010 ± 0.002 | 0.217 ± 0.050 | – |
| VIO + anchors (EKF) | 0.060 ± 0.019 | 0.410 ± 0.058 | 0.030 ± 0.015 | 0.001 ± 0.001 | 0.034 ± 0.010 | – |
| VIO + landmarks (graph SLAM) | 0.119 ± 0.015 | 2.422 ± 0.769 | 0.187 ± 0.065 | 0.009 ± 0.003 | 0.188 ± 0.064 | 0.132 ± 0.038 |

## E3_closed_loop

Detection: {'frames_with_marker_pct': 87.35849056603774, 'longest_marker_free_gap_s': 0.5, 'n_detections': 511, 'detection_success_pct': 89.96478873239437}

| Method | ATE RMSE [m] | Rot RMSE [deg] | End drift [m] | Drift/m | Loop residual [m] | Marker err [m] |
|---|---|---|---|---|---|---|
| IMU only | 25.011 ± 12.532 | 2.205 ± 1.447 | 54.878 ± 27.843 | 2.502 ± 1.269 | 54.882 ± 27.842 | – |
| VIO | 0.118 ± 0.016 | 2.437 ± 1.084 | 0.166 ± 0.047 | 0.008 ± 0.002 | 0.165 ± 0.050 | – |
| VIO + anchors (EKF) | 0.038 ± 0.008 | 0.335 ± 0.056 | 0.040 ± 0.015 | 0.002 ± 0.001 | 0.040 ± 0.017 | – |
| VIO + landmarks (graph SLAM) | 0.072 ± 0.021 | 1.895 ± 0.465 | 0.077 ± 0.034 | 0.004 ± 0.002 | 0.076 ± 0.036 | 0.073 ± 0.026 |

## E5_difficult

Detection: {'frames_with_marker_pct': 15.934065934065933, 'longest_marker_free_gap_s': 3.9, 'n_detections': 58, 'detection_success_pct': 43.609022556390975}

| Method | ATE RMSE [m] | Rot RMSE [deg] | End drift [m] | Drift/m | Loop residual [m] | Marker err [m] |
|---|---|---|---|---|---|---|
| IMU only | 12.393 ± 7.365 | 3.225 ± 0.759 | 29.154 ± 19.366 | 1.336 ± 0.888 | 29.160 ± 19.363 | – |
| VIO | 0.148 ± 0.025 | 2.534 ± 0.868 | 0.169 ± 0.034 | 0.008 ± 0.002 | 0.176 ± 0.034 | – |
| VIO + anchors (EKF) | 0.094 ± 0.024 | 1.124 ± 0.437 | 0.090 ± 0.055 | 0.004 ± 0.003 | 0.091 ± 0.052 | – |
| VIO + landmarks (graph SLAM) | 0.141 ± 0.020 | 4.504 ± 1.629 | 0.163 ± 0.027 | 0.007 ± 0.001 | 0.170 ± 0.028 | 0.175 ± 0.033 |

## E4_relocalisation

Detection: {'frames_with_marker_pct': 52.43243243243243, 'longest_marker_free_gap_s': 3.0, 'n_detections': 120, 'detection_success_pct': 60.301507537688444}

| Method | ATE RMSE [m] | Rot RMSE [deg] | End drift [m] | Drift/m | Reloc time [s] |
|---|---|---|---|---|---|
| Relocalisation (map + EKF) | 0.093 ± 0.013 | 1.514 ± 0.410 | 0.091 ± 0.039 | 0.014 ± 0.006 | 3.000 ± 0.000 |
| VIO | 1.569 ± 0.936 | 57.053 ± 31.406 | 1.073 ± 0.488 | 0.161 ± 0.073 | – |
| EKF + map, wrong init | 1.549 ± 0.956 | 54.923 ± 33.664 | 0.865 ± 0.710 | 0.130 ± 0.106 | – |
