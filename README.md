# my_ekf_pkg — Task 1a (IST Intro to Robotics, Mini-Project 1)

Estimate the pose of a TurtleBot3 Waffle Pi from wheel odometry and IMU with

1. the EKF of `robot_localization` (with and without the 1 Hz ground truth), and
2. **our own EKF**, which fuses odometry and IMU at their full rate and the
   ground truth (GT) downsampled to 1 Hz as a correction,

and evaluate all estimates against the full-rate motion-capture ground truth.

## Package layout

| File | Content |
|---|---|
| `my_ekf_pkg/geometry.py` | angle wrapping, quaternion <-> yaw, 2D pose `compose` / `invert` |
| `my_ekf_pkg/ekf.py` | `UnicycleEKF`: motion model, Jacobian, predict, generic update |
| `my_ekf_pkg/fusion.py` | `FusionConfig` (parameters) and `EkfFusion` (sensor handling, time handling) |
| `my_ekf_pkg/ground_truth.py` | `GroundTruthConverter` (mocap laser -> base_footprint in odom), `RateLimiter` |
| `my_ekf_pkg/metrics.py` | GT interpolation, errors, RMSE, consistency (NEES) |
| `my_ekf_pkg/msg_utils.py` | message <-> pose / covariance helpers shared by nodes and tools |
| `my_ekf_pkg/bag_utils.py` | rosbag2 reading (storage id from `metadata.yaml`) |
| `my_ekf_pkg/ekf_node.py` | ROS node of our EKF (`/ekf/odom`, `/ekf/path`) |
| `my_ekf_pkg/gt_publisher.py` | ROS node publishing the GT (`/gt/pose_full`, `/gt/pose_1hz`, `/gt/path`) |
| `my_ekf_pkg/evaluate.py` | evaluates a recorded results bag (tables, CSV, plots) |
| `my_ekf_pkg/run_offline.py` | runs the same filter directly on the dataset bag (parameter sweeps) |
| `my_ekf_pkg/plot_results.py` | plots the CSV written by `ekf_node` |
| `config/ekf_params.yaml` | parameters of `ekf_node` and `gt_publisher` |
| `config/ekf_rl.yaml` | the two `robot_localization` configurations |
| `launch/task1a.launch.py` | everything together |

The math modules have no ROS imports, so the ROS node and the offline runner
share exactly the same code.

## Ground truth

The bag has no ground-truth topic. The motion capture is in `/tf` as the
transform `mocap -> mocap_laser_link` (60 Hz), and it tracks the **laser**
(`base_scan`), not the robot centre. `gt_publisher` converts every sample to
the pose of `base_footprint` in `odom` (the frame all estimates live in):

```
T_odom_base = T_odom_mocap * T_mocap_laser * inv(T_base_laser)
```

* `T_odom_mocap` is the static transform published by the dataset's
  `publish_initial_tf` (looked up once with tf2),
* `T_base_laser = (-0.064, 0, 0)` is the laser mount of the Waffle Pi
  (check with `ros2 run tf2_ros tf2_echo base_footprint base_scan`).

The composition is done in 2D (x, y, yaw); roll/pitch of the mocap frames is
negligible. At the start the converted GT is ~(0, 0, 0), which the node checks
and logs.

## Our EKF

### State and motion model

State in frame `odom`: **x = [x, y, θ, v, ω]** (position, heading, forward
speed, yaw rate).

Constant-velocity unicycle model for a step `dt`:

```
x_k+1 = x_k + v cos(θ) dt
y_k+1 = y_k + v sin(θ) dt
θ_k+1 = wrap(θ_k + ω dt)
v_k+1 = v_k,   ω_k+1 = ω_k
```

Jacobian `F = ∂f/∂x` (θ before the step), identity plus

```
F[x,θ] = -v sin(θ) dt    F[x,v] = cos(θ) dt
F[y,θ] =  v cos(θ) dt    F[y,v] = sin(θ) dt
F[θ,ω] = dt
```

Prediction: `P = F P Fᵀ + Q`, with **`Q = diag(q_xy, q_xy, q_θ, q_v, q_ω) · dt`**.
The `q_*` are continuous-time intensities (variance added per second), so the
growth of uncertainty depends on elapsed time, not on how many messages arrive.
Long intervals are split into sub-steps of at most `max_step = 0.02 s`.

Why v and ω are in the state: odometry (~25 Hz) and IMU (~100+ Hz) arrive at
different times. With the velocities in the state, every message is a normal
Kalman update and the filter can combine both sensors optimally. The old
"use the last received value as input" approach could not weight the sensors
by their noise.

### Measurement updates

Generic update (`UnicycleEKF.update`):

```
y = z − H x                (angle rows wrapped to (−π, π])
S = H P Hᵀ + R
K = P Hᵀ S⁻¹
x = x + K y                (θ wrapped)
P = (I − K H) P (I − K H)ᵀ + K R Kᵀ   (Joseph form, then symmetrised)
NIS = yᵀ S⁻¹ y
```

| Sensor | z | H | R |
|---|---|---|---|
| `/odom` | `twist.twist.linear.x` (+ `angular.z` if `use_odom_omega`) | row of v (and ω) | `sigma_odom_v²` (`sigma_odom_omega²`) |
| `/imu` | `angular_velocity.z` | row of ω | `sigma_imu_omega²` |
| `/gt/pose_1hz` | x, y, yaw | first 3 rows of I₅ | message covariance (indices 0,1,5 / 6,7,11 / 30,31,35) if valid, else `sigma_gt_*²` |

IMU orientation and accelerations are not used (orientation would be a second
yaw source that drifts, accelerations are too noisy to integrate).

### Time handling

All timing uses `header.stamp` (nodes run with `use_sim_time: true`).

* The first message initialises the filter at its stamp with x₀ = 0 (the
  `odom` frame starts at the robot).
* For every message: predict to its stamp, then update.
* A message that is older than the filter time (odometry and IMU are stamped
  by different clocks) is applied at the current filter time; the filter is
  **never rewound**, so time is never counted twice.
* If a sensor's stamps jump back by more than `time_jump_reset` (bag
  restarted), the filter resets. The check is per sensor so that a constant
  clock offset between two sensors cannot trigger repeated resets.
* At every GT message the error of the *prediction* (before the update) is
  recorded; the update is only applied if `use_gt` is true. So a run without GT
  is evaluated the same way.

### Parameters (`config/ekf_params.yaml`)

| Parameter | Value | Justification |
|---|---|---|
| `sigma_odom_v` | 0.02 m/s | encoder quantisation + slip; std of odom speed when still is a lower bound |
| `sigma_odom_omega` | 0.05 rad/s | wheel-based yaw rate is poor (slip in turns); unused by default |
| `sigma_imu_omega` | 0.02 rad/s | gyro noise, measure as std of `angular_velocity.z` when still |
| `sigma_gt_xy`, `sigma_gt_yaw` | 0.01 m, 0.0175 rad | mocap is mm-accurate; margin for the laser offset / 2D projection |
| `q_xy`, `q_theta` | 1e-4 | small: the model is good, covers slip / lateral motion |
| `q_v` | 0.25 (m/s)²/s | speed can change by ~0.5 m/s in 1 s (TurtleBot accelerations) |
| `q_omega` | 1.0 (rad/s)²/s | yaw rate can change by ~1 rad/s in 1 s |
| `init_sigma_*` | 0.01 / 0.01 / 0.1 / 0.1 | pose known at start (odom origin), velocities less so |
| `max_step` | 0.02 s | keeps the linearisation accurate for long gaps |

**Estimating the sensor sigmas.** The robot stands still at the beginning of
the bag, so the spread of the readings there is pure noise:

```bash
ros2 run my_ekf_pkg run_offline <dataset_bag> --noise-window 3.0
```

prints mean and std of IMU `angular_velocity.z` and odom v, ω over the first
3 s. Use the std as sigma (and the IMU mean as an estimate of the gyro bias).
Check that the robot really does not move in that window (the tool warns). The
odometry often reports exactly 0 while standing, so keep a floor for
`sigma_odom_v`.

## robot_localization (`config/ekf_rl.yaml`)

Two nodes with the same inputs as our EKF: `/odom` → vx, vy (vy = 0 is a
useful non-holonomic constraint), `/imu` → vyaw only (one yaw-rate source).
`ekf_rl_gt` additionally fuses `/gt/pose_1hz` → x, y, yaw. `two_d_mode`,
`publish_tf: false` (the bag already has odom → base_footprint). Notes:
robot_localization clamps measurement variances below 1e-9 (a zero covariance
in a message = "perfect"), multiplies `process_noise_covariance` by dt and
subscribes with SensorDataQoS.

## How to run

```bash
cd ~/ros2_ws
colcon build --packages-select my_ekf_pkg --symlink-install
source install/setup.bash

# everything (bag playback + RViz + our EKF + robot_localization), record results
ros2 launch my_ekf_pkg task1a.launch.py record:=true bag_out:=task1a_results
```

Launch arguments: `play_bag` (true), `viz` (`rviz2` | `foxglove`), `use_rl`
(true), `record` (false), `bag_out` (task1a_results), `params_file`,
`rl_params_file`. The dataset bag path is set in the dataset package's
`turtlebot_playbag.launch.py`.

RViz: Fixed Frame `odom`; add *Path* displays for `/gt/path` and `/ekf/path`
and an *Odometry* display for `/ekf/odom` (enable covariance) and
`/rl/gt/odom`, `/rl/no_gt/odom`.

Quick checks while it runs:

```bash
ros2 topic hz /gt/pose_1hz          # ~1 Hz
ros2 topic hz /ekf/odom             # ~ odom + IMU rate
ros2 topic echo /odom --once        # twist.covariance
ros2 topic echo /imu --once         # angular_velocity_covariance
ros2 run tf2_ros tf2_echo base_footprint base_scan   # laser offset
```

### Evaluation

```bash
ros2 run my_ekf_pkg evaluate task1a_results            # opens plots
ros2 run my_ekf_pkg evaluate task1a_results --no-show  # headless
```

For `/ekf/odom`, `/rl/gt/odom`, `/rl/no_gt/odom` and `/odom`, the GT
(`/gt/pose_full`) is interpolated at every estimate stamp. Printed and saved to
`task1a_results_eval/summary.csv`:

* position error RMSE / mean / max / final, heading RMSE,
* **pre-correction error**: the error just before each 1 Hz GT update (how far
  the filter drifts in 1 s on odometry + IMU only),
* **consistency**: fraction of samples with `[ex ey] P⁻¹ [ex ey]ᵀ ≤ 5.991`
  (inside the 95 % ellipse, χ² with 2 dof; ideal ≈ 0.95) and mean NEES (ideal ≈ 2).
  Much lower fraction → the filter is over-confident (Q or R too small);
  close to 1 with small NEES → too pessimistic.

Plots: XY trajectories, position error vs time, heading error vs time, and the
error with the filter's own `2·sqrt(trace P_xy)` band.

`ekf_node` also writes `~/ros2_ws/ekf_evaluation.csv` on shutdown
(`time,error_dist,err_x,err_y,err_yaw,sigma_pos,nis`, pre-update errors at the
GT instants) which `python3 my_ekf_pkg/plot_results.py` plots.

### Parameter experiments (offline, seconds per run)

```bash
ros2 run my_ekf_pkg run_offline <dataset_bag> --params config/ekf_params.yaml
ros2 run my_ekf_pkg run_offline <dataset_bag> --no-gt                  # odom + IMU only
ros2 run my_ekf_pkg run_offline <dataset_bag> --set q_v=1.0 --set sigma_imu_omega=0.05
ros2 run my_ekf_pkg run_offline <dataset_bag> --set use_imu=false --set use_odom_omega=true
ros2 run my_ekf_pkg run_offline <dataset_bag> --plot --no-show --csv offline.csv
```

It reads `/odom`, `/imu` and `/tf` from the dataset bag, sorts them by header
stamp, converts the GT like `gt_publisher` (static mocap → odom from
`publish_initial_tf`'s values), runs `EkfFusion` and prints the same metrics
plus the GT innovation NIS (mean ≈ 3 and ~95 % below 7.815 for a consistent
filter). Suggested experiments: with / without GT; IMU vs odometry yaw rate;
scaling Q and R by 0.1× / 10×; GT rate 0.5 / 1 / 2 Hz (`--gt-rate`).
