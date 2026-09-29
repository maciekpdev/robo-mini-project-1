"""
Evaluation metrics against full-rate ground truth (pure Python, no ROS).

Estimates and ground truth are sampled at different times, so the ground
truth is linearly interpolated at every estimate stamp (yaw is unwrapped
first, so interpolation across +-pi works).
"""

from dataclasses import dataclass, field
import math

from my_ekf_pkg.geometry import wrap_angle
import numpy as np

CHI2_2DOF_95 = 5.991
CHI2_3DOF_95 = 7.815


@dataclass
class Trajectory:
    """A time series of 2D poses with optional (xx, xy, yy) position covariance."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    cov_xy: np.ndarray = None

    @classmethod
    def from_samples(cls, samples):
        """
        Build from a list of (t, x, y, yaw) or (t, x, y, yaw, cov3) tuples.

        cov3 may be None; the trajectory gets a covariance array only if at
        least one sample has one. Samples are sorted by time.
        """
        samples = sorted(samples, key=lambda s: s[0])
        arr = np.array([s[:4] for s in samples], dtype=float).reshape(-1, 4)
        cov = None
        if any(len(s) > 4 and s[4] is not None for s in samples):
            cov = np.full((len(samples), 3), np.nan)
            for i, s in enumerate(samples):
                if len(s) > 4 and s[4] is not None:
                    c = np.asarray(s[4], dtype=float)
                    cov[i] = (c[0, 0], c[0, 1], c[1, 1])
        return cls(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], cov)

    def __len__(self):
        """Return the number of samples."""
        return len(self.t)


@dataclass
class TrackResult:
    """Per-sample errors and summary statistics of one estimate."""

    t: np.ndarray
    ex: np.ndarray
    ey: np.ndarray
    e_pos: np.ndarray
    e_yaw: np.ndarray
    sigma_pos: np.ndarray
    nees: np.ndarray
    summary: dict = field(default_factory=dict)


def interpolate_poses(gt, t, max_gap=0.2):
    """
    Interpolate ground truth at times t.

    :param gt: Trajectory with the ground truth.
    :param max_gap: query points whose neighbouring GT samples are further
                    apart than this [s] (or outside the GT span) are invalid.
    :return: (x, y, yaw, valid) arrays.
    """
    t = np.asarray(t, dtype=float)
    gt_t, idx = np.unique(gt.t, return_index=True)
    if len(gt_t) < 2:
        nan = np.full(len(t), np.nan)
        return nan, nan.copy(), nan.copy(), np.zeros(len(t), dtype=bool)
    gx, gy = gt.x[idx], gt.y[idx]
    gyaw = np.unwrap(gt.yaw[idx])

    x = np.interp(t, gt_t, gx)
    y = np.interp(t, gt_t, gy)
    yaw = wrap_angle(np.interp(t, gt_t, gyaw))

    valid = (t >= gt_t[0]) & (t <= gt_t[-1])
    hi = np.clip(np.searchsorted(gt_t, t, side='left'), 1, len(gt_t) - 1)
    gap = gt_t[hi] - gt_t[hi - 1]
    valid &= gap <= max_gap
    return x, y, np.atleast_1d(yaw), valid


def position_nees(ex, ey, cov_xy):
    """Return [ex ey] P^-1 [ex ey]^T per sample (NaN where P is unusable)."""
    cxx, cxy, cyy = cov_xy[:, 0], cov_xy[:, 1], cov_xy[:, 2]
    det = cxx * cyy - cxy ** 2
    ok = np.isfinite(det) & (cxx > 0) & (cyy > 0) & (det > 0)
    nees = np.full(len(ex), np.nan)
    a, b = ex[ok], ey[ok]
    quad = cyy[ok] * a * a - 2.0 * cxy[ok] * a * b + cxx[ok] * b * b
    nees[ok] = quad / det[ok]
    return nees


def errors_before(t, e_pos, times, max_gap=0.2):
    """Return the error of the last sample strictly before each of `times`."""
    out = []
    for tc in times:
        i = int(np.searchsorted(t, tc, side='left')) - 1
        if i >= 0 and tc - t[i] <= max_gap:
            out.append(e_pos[i])
    return np.array(out)


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a)))) if len(a) else math.nan


def evaluate_track(est, gt, correction_times=(), max_gap=0.2):
    """
    Compare an estimated Trajectory with the full-rate ground truth.

    :param correction_times: stamps of the 1 Hz GT corrections; the error just
                             before each one measures the drift over 1 s.
    :return: TrackResult (per-sample arrays + ``summary`` dict).
    """
    gx, gy, gyaw, valid = interpolate_poses(gt, est.t, max_gap)
    t = est.t[valid]
    ex = est.x[valid] - gx[valid]
    ey = est.y[valid] - gy[valid]
    e_pos = np.hypot(ex, ey)
    e_yaw = wrap_angle(est.yaw[valid] - gyaw[valid])
    e_yaw = np.atleast_1d(e_yaw)

    if est.cov_xy is not None:
        cov = est.cov_xy[valid]
        sigma_pos = np.sqrt(cov[:, 0] + cov[:, 2])
        nees = position_nees(ex, ey, cov)
    else:
        sigma_pos = np.full(len(t), np.nan)
        nees = np.full(len(t), np.nan)

    before = errors_before(t, e_pos, correction_times, max_gap)
    nees_ok = nees[np.isfinite(nees)]
    summary = {
        'n_samples': int(len(t)),
        'rmse_pos': _rms(e_pos),
        'mean_pos': float(np.mean(e_pos)) if len(t) else math.nan,
        'max_pos': float(np.max(e_pos)) if len(t) else math.nan,
        'final_pos': float(e_pos[-1]) if len(t) else math.nan,
        'rmse_yaw_deg': math.degrees(_rms(e_yaw)),
        'max_yaw_deg': math.degrees(float(np.max(np.abs(e_yaw)))) if len(t) else math.nan,
        'before_corr_mean': float(np.mean(before)) if len(before) else math.nan,
        'before_corr_max': float(np.max(before)) if len(before) else math.nan,
        'consistency_95': (float(np.mean(nees_ok <= CHI2_2DOF_95))
                           if len(nees_ok) else math.nan),
        'mean_nees': float(np.mean(nees_ok)) if len(nees_ok) else math.nan,
    }
    return TrackResult(t, ex, ey, e_pos, e_yaw, sigma_pos, nees, summary)


SUMMARY_COLUMNS = [
    ('n_samples', 'N', '{:d}'),
    ('rmse_pos', 'RMSE[m]', '{:.3f}'),
    ('mean_pos', 'mean[m]', '{:.3f}'),
    ('max_pos', 'max[m]', '{:.3f}'),
    ('final_pos', 'final[m]', '{:.3f}'),
    ('rmse_yaw_deg', 'yawRMSE[deg]', '{:.2f}'),
    ('before_corr_mean', 'pre-corr mean[m]', '{:.3f}'),
    ('before_corr_max', 'pre-corr max[m]', '{:.3f}'),
    ('consistency_95', 'in 95%', '{:.2f}'),
    ('mean_nees', 'mean NEES(2)', '{:.2f}'),
]


def format_summary(summaries):
    """Format {name: summary dict} as a fixed-width text table."""
    name_w = max([len('track')] + [len(n) for n in summaries])
    header = 'track'.ljust(name_w) + ''.join(
        f' | {label:>{max(len(label), 7)}}' for _, label, _ in SUMMARY_COLUMNS)
    lines = [header, '-' * len(header)]
    for name, s in summaries.items():
        cells = []
        for key, label, fmt in SUMMARY_COLUMNS:
            value = s.get(key, math.nan)
            if isinstance(value, float) and math.isnan(value):
                text = 'n/a'
            else:
                text = fmt.format(value)
            cells.append(f' | {text:>{max(len(label), 7)}}')
        lines.append(name.ljust(name_w) + ''.join(cells))
    return '\n'.join(lines)
