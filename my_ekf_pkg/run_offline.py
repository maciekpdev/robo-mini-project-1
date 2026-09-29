"""
Run our EKF directly on the dataset bag, without playback (fast tuning).

Usage::

    ros2 run my_ekf_pkg run_offline <dataset_bag> [--params config/ekf_params.yaml]
        [--set q_v=0.5 --set sigma_imu_omega=0.01] [--no-gt] [--plot]
        [--noise-window 3.0]

The messages /odom, /imu and the ground truth (TF mocap -> mocap_laser_link
in /tf) are read from the bag, sorted by header stamp and fed to the same
EkfFusion class used by ekf_node. The ground truth is converted exactly like
gt_publisher does (static mocap -> odom from publish_initial_tf, laser ->
base_footprint offset) and downsampled to 1 Hz for the corrections; the full
rate ground truth is used for the evaluation.
"""

import argparse
import csv
from dataclasses import fields
import math
import os

from my_ekf_pkg.fusion import CSV_HEADER, EkfFusion, FusionConfig
from my_ekf_pkg.geometry import invert
from my_ekf_pkg.ground_truth import (default_mocap_to_odom, DEFAULT_SENSOR_OFFSET,
                                     GroundTruthConverter, RateLimiter)
from my_ekf_pkg.metrics import CHI2_3DOF_95, evaluate_track, format_summary, Trajectory
from my_ekf_pkg.msg_utils import pose2d_from_pose, pose2d_from_transform, stamp_to_sec
import numpy as np
import yaml

EVENT_ORDER = {'odom': 0, 'imu': 1, 'gt': 2}


def load_params_file(path):
    """
    Read a ROS 2 parameter YAML file.

    :return: (EKF overrides for FusionConfig, gt_publisher parameters)
    """
    with open(os.path.expanduser(path), 'r') as f:
        data = yaml.safe_load(f) or {}

    def section(node):
        for key in (node, '/' + node, '/**'):
            if key in data:
                return data[key].get('ros__parameters', {}) or {}
        return {}

    names = {f.name for f in fields(FusionConfig)}
    ekf = {k: v for k, v in section('ekf_node').items() if k in names}
    return ekf, section('gt_publisher')


def parse_overrides(items):
    """Parse ['name=value', ...] into a dict (values cast later by FusionConfig)."""
    out = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f'--set expects name=value, got {item!r}')
        name, value = item.split('=', 1)
        out[name.strip()] = value.strip()
    return out


def load_dataset(bag, mocap_frame='mocap', gt_frame='mocap_laser_link',
                 odom_topic='/odom', imu_topic='/imu', tf_topic='/tf'):
    """
    Read the sensor data and the raw mocap poses from the dataset bag.

    :return: dict with lists 'odom' (t, v, omega, pose), 'imu' (t, omega) and
             'mocap' (t, laser pose in mocap), each sorted by stamp.
    """
    from my_ekf_pkg.bag_utils import read_messages

    data = {'odom': [], 'imu': [], 'mocap': []}
    for topic, msg, _ in read_messages(bag, [odom_topic, imu_topic, tf_topic]):
        if topic == odom_topic:
            data['odom'].append((stamp_to_sec(msg.header.stamp), msg.twist.twist.linear.x,
                                 msg.twist.twist.angular.z, pose2d_from_pose(msg.pose.pose)))
        elif topic == imu_topic:
            data['imu'].append((stamp_to_sec(msg.header.stamp), msg.angular_velocity.z))
        else:
            for tr in msg.transforms:
                if (tr.header.frame_id.lstrip('/') == mocap_frame
                        and tr.child_frame_id.lstrip('/') == gt_frame):
                    data['mocap'].append((stamp_to_sec(tr.header.stamp),
                                          pose2d_from_transform(tr.transform)))
    for key in data:
        data[key].sort(key=lambda s: s[0])
    return data


def print_noise_stats(data, window):
    """
    Print sensor statistics over the first `window` seconds of the bag.

    If the robot stands still at the start, the standard deviations are
    estimates of the sensor noise (sigma_imu_omega, sigma_odom_v, ...) and the
    mean IMU rate is the gyro bias.
    """
    if not data['odom'] or not data['imu']:
        print('Noise stats: /odom or /imu missing.')
        return
    t0 = min(data['odom'][0][0], data['imu'][0][0])
    v = np.array([s[1] for s in data['odom'] if s[0] <= t0 + window])
    w_odom = np.array([s[2] for s in data['odom'] if s[0] <= t0 + window])
    w_imu = np.array([s[1] for s in data['imu'] if s[0] <= t0 + window])
    print(f'Sensor statistics over the first {window:.1f} s '
          f'({len(v)} odom msgs, {len(w_imu)} IMU msgs):')
    for name, a in (('odom v [m/s]', v), ('odom omega [rad/s]', w_odom),
                    ('imu omega [rad/s]', w_imu)):
        if len(a) > 1:
            print(f'  {name:20s} mean={np.mean(a): .5f}  std={np.std(a, ddof=1):.5f}')
    if len(v) and np.max(np.abs(v)) > 0.01:
        print('  Warning: the robot moves in this window; the std is NOT pure noise.')


def run_filter(config, data, converter, gt_rate, gt_cov=None):
    """
    Run EkfFusion over the dataset.

    :param gt_cov: 3x3 covariance attached to the 1 Hz GT (like gt_publisher).
    :return: (fusion, estimate Trajectory, full-rate GT Trajectory,
              correction times)
    """
    gt_full = [(t, *converter.convert(p)) for t, p in data['mocap']]
    limiter = RateLimiter(gt_rate)
    events = [(t, 'odom', (v, w)) for t, v, w, _ in data['odom']]
    events += [(t, 'imu', w) for t, w in data['imu']]
    corr_times = []
    for t, x, y, yaw in gt_full:
        if limiter.accept(t):
            corr_times.append(t)
            events.append((t, 'gt', (x, y, yaw)))
    events.sort(key=lambda e: (e[0], EVENT_ORDER[e[1]]))

    fusion = EkfFusion(config)
    estimate = []
    for t, kind, payload in events:
        if kind == 'odom':
            fusion.on_odom(t, *payload)
        elif kind == 'imu':
            fusion.on_imu(t, payload)
        else:
            fusion.on_gt(t, payload, gt_cov)
        x, P = fusion.ekf.x, fusion.ekf.P
        estimate.append((fusion.time, x[0], x[1], x[2], P[:2, :2]))

    return (fusion, Trajectory.from_samples(estimate), Trajectory.from_samples(gt_full),
            corr_times)


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('bag', help='dataset bag directory (e.g. fixed_slam_easy)')
    parser.add_argument('--params', default=None,
                        help='ROS parameter YAML (sections ekf_node / gt_publisher)')
    parser.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                        help='override an EKF parameter (repeatable)')
    parser.add_argument('--no-gt', action='store_true',
                        help='do not fuse the 1 Hz ground truth (use_gt=false)')
    parser.add_argument('--gt-rate', type=float, default=None, help='GT rate [Hz]')
    parser.add_argument('--sensor-offset', type=float, nargs=2, default=None,
                        metavar=('DX', 'DY'), help='laser position in base_footprint [m]')
    parser.add_argument('--noise-window', type=float, default=0.0,
                        help='print sensor mean/std over the first N seconds')
    parser.add_argument('--csv', default=None,
                        help='write errors at the GT instants (plot_results format)')
    parser.add_argument('--plot', action='store_true', help='save (and show) plots')
    parser.add_argument('--no-show', action='store_true', help='do not open plot windows')
    parser.add_argument('--out-dir', default='offline_eval', help='directory for plots')
    parser.add_argument('--max-gap', type=float, default=0.2)
    args = parser.parse_args(argv)

    config = FusionConfig()
    gt_params = {}
    if args.params:
        ekf_params, gt_params = load_params_file(args.params)
        config.update(ekf_params)
    config.update(parse_overrides(args.set))
    if args.no_gt:
        config.use_gt = False

    offset = args.sensor_offset or (
        float(gt_params.get('sensor_offset_x', DEFAULT_SENSOR_OFFSET[0])),
        float(gt_params.get('sensor_offset_y', DEFAULT_SENSOR_OFFSET[1])))
    gt_rate = args.gt_rate or float(gt_params.get('rate_hz', 1.0))
    sxy = float(gt_params.get('sigma_xy', 0.01))
    syaw = float(gt_params.get('sigma_yaw', 0.0175))
    gt_cov = np.diag([sxy ** 2, sxy ** 2, syaw ** 2])
    converter = GroundTruthConverter(invert(default_mocap_to_odom()), offset)

    data = load_dataset(args.bag)
    print(f"Read {len(data['odom'])} odom, {len(data['imu'])} IMU and "
          f"{len(data['mocap'])} mocap samples.")
    for key in ('odom', 'imu', 'mocap'):
        s = data[key]
        if len(s) > 1 and s[-1][0] > s[0][0]:
            rate = (len(s) - 1) / (s[-1][0] - s[0][0])
            print(f'  {key:5s}: {rate:7.1f} Hz (by header stamp)')
    if not data['mocap']:
        raise SystemExit('No mocap -> mocap_laser_link transforms found in /tf.')
    if args.noise_window > 0.0:
        print_noise_stats(data, args.noise_window)

    fusion, est, gt, corr_times = run_filter(config, data, converter, gt_rate, gt_cov)
    first = gt.x[0], gt.y[0], math.degrees(gt.yaw[0])
    print('First GT pose in odom: x={:.3f} m, y={:.3f} m, yaw={:.2f} deg'.format(*first))
    print('Parameters: ' + ', '.join(f'{k}={v}' for k, v in config.to_dict().items()))
    if fusion.n_stale:
        print(f'{fusion.n_stale} messages older than the filter time '
              f'(max lag {fusion.max_lag * 1000:.1f} ms) were applied without rewinding.')

    odom_track = Trajectory.from_samples([(t, *p) for t, _, _, p in data['odom']])
    tracks = {'ekf': est, '/odom': odom_track}
    results = {name: evaluate_track(tr, gt, corr_times, args.max_gap)
               for name, tr in tracks.items()}
    print(format_summary({n: r.summary for n, r in results.items()}))

    nis = np.array([r.nis for r in fusion.gt_log])
    if len(nis):
        print(f'GT innovation NIS (3 dof): mean={np.mean(nis):.2f} (ideal 3), '
              f'{100 * np.mean(nis <= CHI2_3DOF_95):.0f}% below {CHI2_3DOF_95} (ideal 95%)')

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(r.as_row() for r in fusion.gt_log)
        print(f'GT-instant errors written to {args.csv}')

    if args.plot:
        from my_ekf_pkg.evaluate import make_plots
        os.makedirs(args.out_dir, exist_ok=True)
        for path in make_plots(tracks, gt, results, args.out_dir, show=not args.no_show,
                               order=['ekf', '/rl/gt/odom', '/rl/no_gt/odom', '/odom']):
            print(f'Plot written to {path}')


if __name__ == '__main__':
    main()
