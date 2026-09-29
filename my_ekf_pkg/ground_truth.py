"""
Ground-truth conversion and downsampling (pure Python, no ROS imports).

The motion-capture ground truth is the TF ``mocap -> mocap_laser_link``: the
pose of the LASER (base_scan) in the ``mocap`` frame. The EKF estimates the
pose of ``base_footprint`` in ``odom``, so the ground truth is converted with::

    T_odom_base = T_odom_mocap * T_mocap_laser * inv(T_base_laser)

where ``T_odom_mocap`` is the static transform published by the dataset's
``publish_initial_tf`` and ``T_base_laser = (dx, dy, 0)`` is the laser mount.
"""

from my_ekf_pkg.geometry import compose, invert, yaw_from_quaternion

# Static transform mocap -> odom published by turtlebot_datasets'
# publish_initial_tf (parent mocap, child odom), i.e. T_mocap_odom.
MOCAP_TO_ODOM_TRANSLATION = (0.935, 1.340, -0.023)
MOCAP_TO_ODOM_QUATERNION = (0.001, -0.003, 0.737, 0.676)  # x, y, z, w

# Laser (base_scan) position in base_footprint for the TurtleBot3 Waffle Pi.
DEFAULT_SENSOR_OFFSET = (-0.064, 0.0)


def default_mocap_to_odom():
    """Return the 2D pose T_mocap_odom from the dataset's static transform."""
    tx, ty, _ = MOCAP_TO_ODOM_TRANSLATION
    return (tx, ty, yaw_from_quaternion(*MOCAP_TO_ODOM_QUATERNION))


class GroundTruthConverter:
    """Convert mocap laser poses into base_footprint poses in odom."""

    def __init__(self, T_odom_mocap=None, sensor_offset=DEFAULT_SENSOR_OFFSET):
        """
        Create the converter.

        :param T_odom_mocap: pose of the mocap frame in odom (x, y, yaw), or
                             None until it is known (see set_odom_mocap).
        :param sensor_offset: (dx, dy) of the tracked laser in base_footprint.
        """
        self.T_odom_mocap = T_odom_mocap
        dx, dy = sensor_offset
        self.T_laser_base = invert((dx, dy, 0.0))

    @property
    def ready(self):
        """Return True once T_odom_mocap is known."""
        return self.T_odom_mocap is not None

    def set_odom_mocap(self, T_odom_mocap):
        """Set the static transform T_odom_mocap (x, y, yaw)."""
        self.T_odom_mocap = tuple(T_odom_mocap)

    def convert(self, T_mocap_laser):
        """Return T_odom_base for a laser pose T_mocap_laser (x, y, yaw)."""
        if not self.ready:
            raise RuntimeError('T_odom_mocap is not set')
        return compose(compose(self.T_odom_mocap, T_mocap_laser), self.T_laser_base)


class RateLimiter:
    """
    Downsample a message stream to a fixed rate using message timestamps.

    Samples are accepted on a fixed grid (t0, t0 + period, ...), so the mean
    spacing is exactly one period even if the input rate is not a multiple of
    the output rate. A backwards jump in time (bag restart) resets the grid.
    """

    def __init__(self, rate_hz, jump_reset=1.0):
        """Create a limiter with output rate rate_hz [Hz]."""
        if rate_hz <= 0.0:
            raise ValueError('rate_hz must be positive')
        self.period = 1.0 / rate_hz
        self.jump_reset = jump_reset
        self.reset()

    def reset(self):
        """Forget the grid; the next sample is accepted."""
        self._next = None
        self._last_t = None

    def accept(self, t):
        """Return True if the sample stamped t should be forwarded."""
        if self._last_t is not None and t < self._last_t - self.jump_reset:
            self.reset()
        if self._last_t is None or t > self._last_t:
            self._last_t = t

        if self._next is None:
            self._next = t + self.period
            return True
        if t >= self._next - 1e-9:
            self._next += self.period
            if self._next <= t:
                # Input had a gap longer than one period: restart the grid.
                self._next = t + self.period
            return True
        return False
