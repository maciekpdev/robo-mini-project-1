"""
Evaluate recorded estimates against the full-rate ground truth.

Usage::

    ros2 run my_ekf_pkg evaluate task1a_results [--no-show] [--out-dir DIR]

Reads the results bag recorded by ``task1a.launch.py record:=true``. For
each estimate topic (default /ekf/odom, /rl/gt/odom, /rl/no_gt/odom, /odom)
the ground truth /gt/pose_full is interpolated at the estimate stamps and
the position / heading errors, the drift just before each 1 Hz correction
and the covariance consistency are computed. A summary CSV and plots are
written to the output directory.
"""

import argparse
import csv
import os

from my_ekf_pkg.metrics import (evaluate_track, format_summary, SUMMARY_COLUMNS,
                                Trajectory)
from my_ekf_pkg.msg_utils import pose_and_cov_from_msg, stamp_to_sec
import numpy as np

DEFAULT_TOPICS = ['/ekf/odom', '/rl/gt/odom', '/rl/no_gt/odom', '/odom']
TRACK_LABELS = {
    '/ekf/odom': 'our EKF (odom+IMU+GT 1 Hz)',
    '/rl/gt/odom': 'robot_localization (odom+IMU+GT 1 Hz)',
    '/rl/no_gt/odom': 'robot_localization (odom+IMU)',
    '/odom': 'raw wheel odometry',
    'ekf': 'our EKF',
}

# Categorical colours in a fixed order: a track keeps its colour (and line
# style, the secondary encoding for print / colour-blind readers) in every plot.
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
SERIES_STYLES = ['-', '--', '-.', ':', (0, (5, 1, 1, 1)), (0, (1, 2))]
GT_COLOR = '#2b2b2b'
GRID_COLOR = '#d9d9d9'


def load_results_bag(path, topics, gt_topic, corr_topic):
    """
    Read estimate tracks, the full-rate GT and the correction times from a bag.

    :return: (tracks {topic: Trajectory}, gt Trajectory, correction times)
    """
    from my_ekf_pkg.bag_utils import read_messages

    samples = {topic: [] for topic in topics}
    gt, corr_times = [], []
    wanted = set(topics) | {gt_topic, corr_topic}
    for topic, msg, _ in read_messages(path, wanted):
        t = stamp_to_sec(msg.header.stamp)
        if topic == gt_topic:
            (x, y, yaw), _ = pose_and_cov_from_msg(msg)
            gt.append((t, x, y, yaw))
        if topic == corr_topic:
            corr_times.append(t)
        if topic in samples:
            (x, y, yaw), cov = pose_and_cov_from_msg(msg)
            samples[topic].append((t, x, y, yaw, cov))
    if len(gt) < 2:
        raise RuntimeError(f'Ground truth topic {gt_topic} not found (or empty) in {path}')
    tracks = {topic: Trajectory.from_samples(s) for topic, s in samples.items() if s}
    for topic in topics:
        if topic not in tracks:
            print(f'Warning: topic {topic} not found in the bag, skipped.')
    return tracks, Trajectory.from_samples(gt), sorted(corr_times)


def evaluate_all(tracks, gt, corr_times, max_gap=0.2):
    """Return {name: TrackResult} for every track."""
    return {name: evaluate_track(tr, gt, corr_times, max_gap) for name, tr in tracks.items()}


def write_summary_csv(results, path):
    """Write one row of summary statistics per track."""
    keys = [key for key, _, _ in SUMMARY_COLUMNS]
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['track'] + keys)
        for name, res in results.items():
            writer.writerow([name] + [res.summary.get(k, '') for k in keys])


def _style(index):
    return {'color': SERIES_COLORS[index % len(SERIES_COLORS)],
            'linestyle': SERIES_STYLES[index % len(SERIES_STYLES)]}


def _label(name):
    return TRACK_LABELS.get(name, name)


def _finish(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left')
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def make_plots(tracks, gt, results, out_dir, show=True, order=None):
    """
    Save XY / error / uncertainty plots as PNG files in out_dir.

    :param order: list of track names that fixes the colour of each track.
    :return: list of written file paths.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    order = list(order) if order else list(tracks)
    index = {name: order.index(name) if name in order else len(order) + i
             for i, name in enumerate(tracks)}
    t0 = gt.t[0]
    written = []

    def save(fig, name):
        fig.tight_layout()
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=200)
        written.append(path)

    # 1) XY trajectories
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(gt.x, gt.y, color=GT_COLOR, linewidth=2.0, label='ground truth (mocap)')
    for name, tr in tracks.items():
        ax.plot(tr.x, tr.y, linewidth=1.5, label=_label(name), **_style(index[name]))
    ax.plot(gt.x[0], gt.y[0], 'o', color=GT_COLOR, markersize=8, label='start')
    ax.set_aspect('equal', adjustable='datalim')
    _finish(ax, 'x [m] (odom)', 'y [m] (odom)', 'Trajectories in frame odom')
    ax.legend(loc='best', fontsize=8, frameon=False)
    save(fig, 'trajectories_xy.png')

    # 2) Position error vs time
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, res in results.items():
        ax.plot(res.t - t0, res.e_pos, linewidth=1.2, label=_label(name),
                **_style(index[name]))
    _finish(ax, 'time [s]', 'position error [m]', 'Position error vs full-rate ground truth')
    ax.set_ylim(bottom=0.0)
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    save(fig, 'position_error.png')

    # 3) Heading error vs time
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, res in results.items():
        ax.plot(res.t - t0, np.degrees(res.e_yaw), linewidth=1.2, label=_label(name),
                **_style(index[name]))
    ax.axhline(0.0, color=GT_COLOR, linewidth=0.8)
    _finish(ax, 'time [s]', 'heading error [deg]', 'Heading error vs full-rate ground truth')
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    save(fig, 'heading_error.png')

    # 4) Error with the filter's own 2-sigma bound (only tracks with covariance)
    with_cov = [n for n, r in results.items() if np.any(np.isfinite(r.sigma_pos))]
    if with_cov:
        fig, axes = plt.subplots(len(with_cov), 1, figsize=(10, 3.2 * len(with_cov)),
                                 sharex=True, squeeze=False)
        for ax, name in zip(axes[:, 0], with_cov):
            res, style = results[name], _style(index[name])
            ts = res.t - t0
            ax.fill_between(ts, 0.0, 2.0 * res.sigma_pos, color=style['color'], alpha=0.2,
                            linewidth=0, label='2 sqrt(trace P_xy)')
            ax.plot(ts, res.e_pos, color=style['color'], linewidth=1.2,
                    label='position error')
            cons = res.summary.get('consistency_95', float('nan'))
            _finish(ax, '', 'error [m]',
                    f'{_label(name)} - {100 * cons:.0f}% of samples inside 95% ellipse')
            ax.set_ylim(bottom=0.0)
            ax.legend(loc='upper left', fontsize=8, frameon=False)
        axes[-1, 0].set_xlabel('time [s]')
        save(fig, 'error_with_uncertainty.png')

    if show:
        plt.show()
    plt.close('all')
    return written


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('bag', help='results bag directory (from record:=true)')
    parser.add_argument('--topics', nargs='+', default=DEFAULT_TOPICS,
                        help='estimate topics to evaluate')
    parser.add_argument('--gt-topic', default='/gt/pose_full')
    parser.add_argument('--corr-topic', default='/gt/pose_1hz',
                        help='1 Hz correction topic (for the pre-correction drift)')
    parser.add_argument('--out-dir', default=None,
                        help='output directory (default: <bag>_eval)')
    parser.add_argument('--max-gap', type=float, default=0.2,
                        help='max GT gap [s] allowed for interpolation')
    parser.add_argument('--no-show', action='store_true', help='do not open plot windows')
    args = parser.parse_args(argv)

    out_dir = args.out_dir or os.path.abspath(args.bag.rstrip('/')) + '_eval'
    os.makedirs(out_dir, exist_ok=True)

    tracks, gt, corr_times = load_results_bag(args.bag, args.topics, args.gt_topic,
                                              args.corr_topic)
    results = evaluate_all(tracks, gt, corr_times, args.max_gap)
    print(f'Ground truth: {len(gt)} samples over {gt.t[-1] - gt.t[0]:.1f} s, '
          f'{len(corr_times)} corrections on {args.corr_topic}')
    print(format_summary({n: r.summary for n, r in results.items()}))

    summary_csv = os.path.join(out_dir, 'summary.csv')
    write_summary_csv(results, summary_csv)
    print(f'Summary written to {summary_csv}')
    for path in make_plots(tracks, gt, results, out_dir, show=not args.no_show,
                           order=args.topics):
        print(f'Plot written to {path}')


if __name__ == '__main__':
    main()
