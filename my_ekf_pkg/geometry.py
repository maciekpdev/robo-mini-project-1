"""
2D geometry helpers (pure Python, no ROS imports).

A 2D pose is a tuple ``(x, y, yaw)``. ``T_A_B`` denotes the pose of frame B
expressed in frame A, i.e. it converts coordinates from B into A:
``p_A = T_A_B * p_B``. Composition follows the usual chain rule:
``T_A_C = compose(T_A_B, T_B_C)``.
"""

import math

import numpy as np


def wrap_angle(angle):
    """Wrap an angle (scalar or numpy array) to the interval (-pi, pi]."""
    wrapped = np.arctan2(np.sin(angle), np.cos(angle))
    if np.ndim(wrapped) == 0:
        return float(wrapped)
    return wrapped


def yaw_from_quaternion(x, y, z, w):
    """Return the yaw (rotation about z) of a quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_from_yaw(yaw):
    """Return the quaternion (x, y, z, w) of a pure rotation about z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def compose(a, b):
    """Return T_A_C = a * b, where a = T_A_B and b = T_B_C."""
    ax, ay, ath = a
    bx, by, bth = b
    c, s = math.cos(ath), math.sin(ath)
    return (ax + c * bx - s * by,
            ay + s * bx + c * by,
            wrap_angle(ath + bth))


def invert(a):
    """Return T_B_A given a = T_A_B."""
    ax, ay, ath = a
    c, s = math.cos(ath), math.sin(ath)
    return (-c * ax - s * ay,
            s * ax - c * ay,
            wrap_angle(-ath))
