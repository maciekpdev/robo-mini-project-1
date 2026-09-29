"""
Extended Kalman Filter for a planar unicycle robot (pure Python, no ROS).

State (frame ``odom``)::

    x = [x, y, theta, v, omega]

``v`` is the forward speed and ``omega`` the yaw rate. Both are modelled as
constant between messages and are corrected by the odometry / IMU
measurements; the pose is obtained by integrating them (prediction step) and
corrected by the 1 Hz ground-truth pose.
"""

from dataclasses import dataclass
import math

from my_ekf_pkg.geometry import wrap_angle
import numpy as np

IX, IY, ITH, IV, IW = range(5)
STATE_DIM = 5


@dataclass
class UpdateResult:
    """Diagnostics of one measurement update."""

    innovation: np.ndarray
    S: np.ndarray
    nis: float


def motion_model(x, dt):
    """Propagate state x by dt with the constant-velocity unicycle model."""
    th, v, w = x[ITH], x[IV], x[IW]
    out = np.array(x, dtype=float)
    out[IX] += v * math.cos(th) * dt
    out[IY] += v * math.sin(th) * dt
    out[ITH] = wrap_angle(th + w * dt)
    return out


def motion_jacobian(x, dt):
    """Return F = d(motion_model)/dx evaluated at x (theta BEFORE the step)."""
    th, v = x[ITH], x[IV]
    c, s = math.cos(th), math.sin(th)
    F = np.eye(STATE_DIM)
    F[IX, ITH] = -v * s * dt
    F[IX, IV] = c * dt
    F[IY, ITH] = v * c * dt
    F[IY, IV] = s * dt
    F[ITH, IW] = dt
    return F


class UnicycleEKF:
    """EKF with state [x, y, theta, v, omega] and a unicycle motion model."""

    def __init__(self, x0, P0, q, max_step=0.02):
        """
        Create the filter.

        :param x0: initial state (5,)
        :param P0: initial covariance (5, 5)
        :param q: process noise intensities (5,) [variance growth per second]
                  for x, y, theta, v, omega. Q = diag(q) * dt.
        :param max_step: longest single prediction step [s]; longer
                         intervals are split into equal sub-steps.
        """
        self.x = np.array(x0, dtype=float).reshape(STATE_DIM)
        self.P = np.array(P0, dtype=float).reshape(STATE_DIM, STATE_DIM)
        self.q = np.array(q, dtype=float).reshape(STATE_DIM)
        self.max_step = float(max_step)

    def predict(self, dt):
        """Propagate state and covariance forward by dt seconds."""
        if dt <= 0.0:
            return
        n_steps = max(1, int(math.ceil(dt / self.max_step - 1e-9)))
        h = dt / n_steps
        Q = np.diag(self.q * h)
        for _ in range(n_steps):
            F = motion_jacobian(self.x, h)
            self.x = motion_model(self.x, h)
            self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def innovation(self, z, H, R, angle_rows=()):
        """Return (y, S) for measurement z without changing the filter."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        H = np.atleast_2d(np.asarray(H, dtype=float))
        R = np.atleast_2d(np.asarray(R, dtype=float))
        y = z - H @ self.x
        for row in angle_rows:
            y[row] = wrap_angle(y[row])
        S = H @ self.P @ H.T + R
        return y, S

    def update(self, z, H, R, angle_rows=()):
        """
        Apply a linear(ised) measurement update z = H x + noise(R).

        :param angle_rows: rows of z that are angles; their innovation is
                           wrapped to (-pi, pi].
        :return: UpdateResult with innovation, S and the NIS y^T S^-1 y.
        """
        H = np.atleast_2d(np.asarray(H, dtype=float))
        R = np.atleast_2d(np.asarray(R, dtype=float))
        y, S = self.innovation(z, H, R, angle_rows)

        # K = P H^T S^-1 (solve instead of explicit inverse; S, P symmetric)
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        self.x[ITH] = wrap_angle(self.x[ITH])

        # Joseph form keeps P symmetric positive definite despite rounding
        I_KH = np.eye(STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        nis = float(y @ np.linalg.solve(S, y))
        return UpdateResult(innovation=y, S=S, nis=nis)
