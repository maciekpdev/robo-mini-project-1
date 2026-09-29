"""
Sensor fusion logic around UnicycleEKF (pure Python, no ROS imports).

``EkfFusion`` receives already-parsed measurements with their timestamps
(``header.stamp`` in seconds) and takes care of initialisation, time handling
and building the measurement models. The ROS node and the offline runner both
use this class, so they produce identical results.

Time handling (messages from different sensors are stamped by different
clocks and may arrive slightly out of order):

* The first message initialises the filter at its stamp with x0 = 0 (the
  ``odom`` frame starts at the robot).
* For every message: predict to the message stamp, then update.
* A message older than the filter time is applied at the current filter time
  (the filter is never rewound, so time is never counted twice).
* If one sensor's stamps jump back by more than ``time_jump_reset`` seconds
  (the bag was restarted), the filter is reset.
"""

from dataclasses import asdict, dataclass, fields
import math

from my_ekf_pkg.ekf import IV, IW, STATE_DIM, UnicycleEKF
from my_ekf_pkg.geometry import wrap_angle
import numpy as np

CSV_HEADER = ['time', 'error_dist', 'err_x', 'err_y', 'err_yaw', 'sigma_pos', 'nis']


@dataclass
class FusionConfig:
    """Filter parameters. Sigmas are standard deviations; q_* are intensities."""

    # Which measurements are fused
    use_odom_v: bool = True
    use_odom_omega: bool = False
    use_imu: bool = True
    use_gt: bool = True

    # Measurement noise (standard deviations)
    sigma_odom_v: float = 0.02         # [m/s]
    sigma_odom_omega: float = 0.05     # [rad/s]
    sigma_imu_omega: float = 0.02      # [rad/s]
    gt_use_msg_covariance: bool = True
    sigma_gt_xy: float = 0.01          # [m]
    sigma_gt_yaw: float = 0.0175       # [rad] (~1 deg)

    # Process noise intensities: variance added per second of prediction
    q_xy: float = 1e-4                 # [m^2/s]
    q_theta: float = 1e-4              # [rad^2/s]
    q_v: float = 0.25                  # [(m/s)^2/s]
    q_omega: float = 1.0               # [(rad/s)^2/s]

    # Initial uncertainty (standard deviations)
    init_sigma_xy: float = 0.01
    init_sigma_theta: float = 0.01
    init_sigma_v: float = 0.1
    init_sigma_omega: float = 0.1

    # Numerics / time handling
    max_step: float = 0.02             # [s] longest prediction sub-step
    time_jump_reset: float = 1.0       # [s] backwards jump that resets

    @classmethod
    def from_dict(cls, values, strict=True):
        """
        Build a config from a dict, casting values to the field types.

        Accepts ints where floats are expected and 'true'/'false' strings for
        booleans. Unknown keys raise KeyError if ``strict``, else are ignored.
        """
        config = cls()
        config.update(values, strict=strict)
        return config

    def update(self, values, strict=True):
        """Override fields from a dict (see from_dict)."""
        types = {f.name: f.type for f in fields(self)}
        for name, value in values.items():
            if name not in types:
                if strict:
                    raise KeyError(f'Unknown EKF parameter: {name}')
                continue
            setattr(self, name, _cast(value, types[name]))

    def to_dict(self):
        """Return the parameters as a plain dict."""
        return asdict(self)

    def process_noise(self):
        """Return the process noise intensities [x, y, theta, v, omega]."""
        return np.array([self.q_xy, self.q_xy, self.q_theta, self.q_v, self.q_omega])

    def initial_covariance(self):
        """Return the initial covariance P0."""
        sig = np.array([self.init_sigma_xy, self.init_sigma_xy, self.init_sigma_theta,
                        self.init_sigma_v, self.init_sigma_omega])
        return np.diag(sig ** 2)

    def gt_noise(self):
        """Return the default 3x3 ground-truth covariance from the sigmas."""
        return np.diag([self.sigma_gt_xy ** 2, self.sigma_gt_xy ** 2, self.sigma_gt_yaw ** 2])


def _cast(value, typ):
    """Cast a parameter value to bool or float."""
    if typ is bool:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ('true', '1', 'yes', 'on'):
                return True
            if lowered in ('false', '0', 'no', 'off'):
                return False
            raise ValueError(f'Not a boolean: {value!r}')
        return bool(value)
    return typ(value)


@dataclass
class GtRecord:
    """Pre-update error of the estimate at one ground-truth instant."""

    time: float
    error_dist: float
    err_x: float
    err_y: float
    err_yaw: float
    sigma_pos: float
    nis: float

    def as_row(self):
        """Return the values in CSV_HEADER order."""
        return [self.time, self.error_dist, self.err_x, self.err_y,
                self.err_yaw, self.sigma_pos, self.nis]


def gt_covariance_is_valid(cov3):
    """Return True if cov3 is a usable 3x3 covariance (finite, positive definite)."""
    if cov3 is None:
        return False
    cov3 = np.asarray(cov3, dtype=float)
    if cov3.shape != (3, 3) or not np.all(np.isfinite(cov3)):
        return False
    if np.any(np.diag(cov3) <= 1e-12):
        return False
    try:
        np.linalg.cholesky(0.5 * (cov3 + cov3.T))
    except np.linalg.LinAlgError:
        return False
    return True


class EkfFusion:
    """Feeds odometry, IMU and ground-truth measurements into a UnicycleEKF."""

    def __init__(self, config=None):
        """Create an uninitialised fusion with the given FusionConfig."""
        self.config = config if config is not None else FusionConfig()
        self.ekf = None
        self.time = None
        self.gt_log = []
        self.n_resets = 0
        self.n_stale = 0
        self.max_lag = 0.0
        self._last_stamp = {}

    # ------------------------------------------------------------------ state
    @property
    def initialized(self):
        """Return True once the first message has been received."""
        return self.ekf is not None

    @property
    def state(self):
        """Return a copy of the state [x, y, theta, v, omega]."""
        return self.ekf.x.copy()

    @property
    def covariance(self):
        """Return a copy of the 5x5 covariance."""
        return self.ekf.P.copy()

    def reset(self):
        """Forget everything (used when the bag restarts)."""
        self.ekf = None
        self.time = None
        self.gt_log = []
        self._last_stamp = {}

    def _initialize(self, t):
        cfg = self.config
        self.ekf = UnicycleEKF(np.zeros(STATE_DIM), cfg.initial_covariance(),
                               cfg.process_noise(), cfg.max_step)
        self.time = t

    def _advance(self, source, t):
        """Handle time for a message from `source` stamped t, then predict."""
        last = self._last_stamp.get(source)
        if last is not None and t < last - self.config.time_jump_reset:
            self.reset()
            self.n_resets += 1
        self._last_stamp[source] = t

        if self.ekf is None:
            self._initialize(t)
            return
        if t > self.time:
            self.ekf.predict(t - self.time)
            self.time = t
        elif t < self.time:
            # Slightly old message: use it at the current time, do not rewind.
            self.n_stale += 1
            self.max_lag = max(self.max_lag, self.time - t)

    # ----------------------------------------------------------- measurements
    def on_odom(self, t, v, omega):
        """Fuse wheel odometry: forward speed v and (optionally) yaw rate omega."""
        self._advance('odom', t)
        cfg = self.config
        rows, z, var = [], [], []
        if cfg.use_odom_v:
            rows.append(IV)
            z.append(v)
            var.append(cfg.sigma_odom_v ** 2)
        if cfg.use_odom_omega:
            rows.append(IW)
            z.append(omega)
            var.append(cfg.sigma_odom_omega ** 2)
        if rows:
            self.ekf.update(z, np.eye(STATE_DIM)[rows], np.diag(var))

    def on_imu(self, t, omega):
        """Fuse the gyroscope yaw rate omega (angular_velocity.z)."""
        self._advance('imu', t)
        if self.config.use_imu:
            self.ekf.update([omega], np.eye(STATE_DIM)[[IW]],
                            [[self.config.sigma_imu_omega ** 2]])

    def on_gt(self, t, pose, cov3=None):
        """
        Handle a (1 Hz) ground-truth pose (x, y, yaw) in frame odom.

        The error of the prediction (before the update) is always recorded in
        ``gt_log``; the update itself is applied only if ``use_gt`` is true.

        :param cov3: optional 3x3 covariance of (x, y, yaw) from the message.
        :return: the GtRecord for this instant.
        """
        self._advance('gt', t)
        cfg = self.config
        if cfg.gt_use_msg_covariance and gt_covariance_is_valid(cov3):
            R = np.asarray(cov3, dtype=float)
        else:
            R = cfg.gt_noise()
        z = np.asarray(pose, dtype=float)
        H = np.eye(STATE_DIM)[:3]

        x, P = self.ekf.x, self.ekf.P
        y, S = self.ekf.innovation(z, H, R, angle_rows=(2,))
        err_x = float(x[0] - z[0])
        err_y = float(x[1] - z[1])
        record = GtRecord(
            time=float(t),
            error_dist=math.hypot(err_x, err_y),
            err_x=err_x,
            err_y=err_y,
            err_yaw=wrap_angle(x[2] - z[2]),
            sigma_pos=math.sqrt(P[0, 0] + P[1, 1]),
            nis=float(y @ np.linalg.solve(S, y)),
        )
        self.gt_log.append(record)

        if cfg.use_gt:
            self.ekf.update(z, H, R, angle_rows=(2,))
        return record
