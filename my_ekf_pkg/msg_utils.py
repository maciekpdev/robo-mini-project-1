"""
Helpers to read/write ROS message fields (duck-typed, no rclpy imports).

Shared by the ROS nodes and the offline tools so both interpret the messages
in exactly the same way.
"""

import math

from my_ekf_pkg.geometry import yaw_from_quaternion
import numpy as np

# Indices of x, y, yaw inside a row-major 6x6 covariance (x y z roll pitch yaw)
COV6_IDX = (0, 1, 5)


def stamp_to_sec(stamp):
    """Convert a builtin_interfaces/Time to float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


def sec_to_stamp_fields(t):
    """Split float seconds into (sec, nanosec) for builtin_interfaces/Time."""
    sec = int(math.floor(t))
    nanosec = int(round((t - sec) * 1e9))
    if nanosec >= 1000000000:
        sec += 1
        nanosec -= 1000000000
    return sec, nanosec


def pose2d_from_pose(pose):
    """Return (x, y, yaw) from a geometry_msgs/Pose."""
    q = pose.orientation
    return (pose.position.x, pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))


def pose2d_from_transform(transform):
    """Return (x, y, yaw) from a geometry_msgs/Transform."""
    q = transform.rotation
    return (transform.translation.x, transform.translation.y,
            yaw_from_quaternion(q.x, q.y, q.z, q.w))


def cov3_from_cov6(cov36):
    """Extract the 3x3 (x, y, yaw) block from a 36-element 6x6 covariance."""
    c = np.asarray(cov36, dtype=float).reshape(6, 6)
    return c[np.ix_(COV6_IDX, COV6_IDX)]


def cov6_from_cov3(cov3):
    """Embed a 3x3 (x, y, yaw) covariance into a 36-element list."""
    c = np.zeros((6, 6))
    c[np.ix_(COV6_IDX, COV6_IDX)] = np.asarray(cov3, dtype=float)
    return [float(v) for v in c.ravel()]


def twist_cov6(var_v, var_w, cov_vw):
    """Return a 36-element twist covariance with only vx and wz filled."""
    c = np.zeros((6, 6))
    c[0, 0] = var_v
    c[5, 5] = var_w
    c[0, 5] = c[5, 0] = cov_vw
    return [float(v) for v in c.ravel()]


def pose_and_cov_from_msg(msg):
    """
    Return ((x, y, yaw), cov3 or None) for pose-like messages.

    Works for nav_msgs/Odometry and geometry_msgs/PoseWithCovarianceStamped
    (``msg.pose.pose`` + ``msg.pose.covariance``) and geometry_msgs/PoseStamped
    (``msg.pose``, no covariance).
    """
    if hasattr(msg.pose, 'pose'):
        return pose2d_from_pose(msg.pose.pose), cov3_from_cov6(msg.pose.covariance)
    return pose2d_from_pose(msg.pose), None
