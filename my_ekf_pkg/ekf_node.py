"""
ROS 2 node running our own EKF (Task 1a).

Subscribes to wheel odometry (/odom), the IMU (/imu) and the 1 Hz ground
truth (/gt/pose_1hz, from gt_publisher) and publishes the estimate as
nav_msgs/Odometry on /ekf/odom plus a nav_msgs/Path on /ekf/path, both in
frame ``odom``. It does NOT broadcast TF (the bag already contains
odom -> base_footprint).

All timing uses message header stamps; run with use_sim_time:=true.
"""

import csv
from dataclasses import fields
import os

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from my_ekf_pkg.fusion import CSV_HEADER, EkfFusion, FusionConfig
from my_ekf_pkg.geometry import quaternion_from_yaw
from my_ekf_pkg.msg_utils import (cov3_from_cov6, cov6_from_cov3, pose2d_from_pose,
                                  sec_to_stamp_fields, stamp_to_sec, twist_cov6)
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

# Node parameters that are not part of FusionConfig
NODE_PARAMS = {
    'odom_topic': '/odom',
    'imu_topic': '/imu',
    'gt_topic': '/gt/pose_1hz',
    'output_topic': '/ekf/odom',
    'path_topic': '/ekf/path',
    'frame_id': 'odom',
    'child_frame_id': 'base_footprint',
    'path_period': 0.1,
    'eval_csv': '~/ros2_ws/ekf_evaluation.csv',
}


class EkfNode(Node):
    """Thin ROS wrapper around EkfFusion."""

    def __init__(self):
        """Declare parameters, create the filter, publishers and subscribers."""
        super().__init__('ekf_node')

        # dynamic_typing lets the YAML contain 1 instead of 1.0 (we cast below)
        any_type = ParameterDescriptor(dynamic_typing=True)
        defaults = FusionConfig()
        for f in fields(FusionConfig):
            self.declare_parameter(f.name, getattr(defaults, f.name), any_type)
        for name, value in NODE_PARAMS.items():
            self.declare_parameter(name, value, any_type)

        config = FusionConfig.from_dict(
            {f.name: self.get_parameter(f.name).value for f in fields(FusionConfig)})
        self.fusion = EkfFusion(config)
        p = {name: self.get_parameter(name).value for name in NODE_PARAMS}
        self.frame_id = str(p['frame_id'])
        self.child_frame_id = str(p['child_frame_id'])
        self.path_period = float(p['path_period'])
        self.eval_csv = os.path.expanduser(str(p['eval_csv']))

        self.path = Path()
        self.path.header.frame_id = self.frame_id
        self.last_path_time = None
        self.n_resets_seen = 0
        self.lag_warned = False

        self.odom_pub = self.create_publisher(Odometry, str(p['output_topic']), 10)
        self.path_pub = self.create_publisher(Path, str(p['path_topic']), 10)
        self.create_subscription(Odometry, str(p['odom_topic']), self.odom_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Imu, str(p['imu_topic']), self.imu_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, str(p['gt_topic']),
                                 self.gt_callback, 10)

        self.get_logger().info(
            'EKF node started with parameters: '
            + ', '.join(f'{k}={v}' for k, v in config.to_dict().items()))

    # -------------------------------------------------------------- callbacks
    def odom_callback(self, msg):
        """Fuse forward speed (and optionally yaw rate) from wheel odometry."""
        self.fusion.on_odom(stamp_to_sec(msg.header.stamp),
                            msg.twist.twist.linear.x, msg.twist.twist.angular.z)
        self.after_measurement()

    def imu_callback(self, msg):
        """Fuse the gyroscope yaw rate."""
        self.fusion.on_imu(stamp_to_sec(msg.header.stamp), msg.angular_velocity.z)
        self.after_measurement()

    def gt_callback(self, msg):
        """Record the pre-update error and (if use_gt) correct with the 1 Hz GT."""
        record = self.fusion.on_gt(stamp_to_sec(msg.header.stamp),
                                   pose2d_from_pose(msg.pose.pose),
                                   cov3_from_cov6(msg.pose.covariance))
        self.get_logger().debug(
            f'GT t={record.time:.2f}: err={record.error_dist:.3f} m, '
            f'sigma={record.sigma_pos:.3f} m, NIS={record.nis:.2f}')
        self.after_measurement()

    # ------------------------------------------------------------- publishing
    def after_measurement(self):
        """Handle resets/diagnostics and publish the current estimate."""
        if self.fusion.n_resets != self.n_resets_seen:
            self.n_resets_seen = self.fusion.n_resets
            self.get_logger().warn('Time jumped back (bag restarted?): EKF reset.')
            self.path.poses = []
            self.last_path_time = None
        if not self.lag_warned and self.fusion.max_lag > self.fusion.config.time_jump_reset:
            self.lag_warned = True
            self.get_logger().warn(
                f'Messages arrive up to {self.fusion.max_lag:.2f} s older than the filter '
                'time; the sensors may be stamped by unsynchronised clocks.')
        self.publish_estimate()

    def make_stamp(self):
        """Return the filter time as a builtin_interfaces/Time."""
        sec, nanosec = sec_to_stamp_fields(self.fusion.time)
        return Time(sec=sec, nanosec=nanosec)

    def publish_estimate(self):
        """Publish /ekf/odom and (every path_period) /ekf/path."""
        x, P = self.fusion.state, self.fusion.covariance
        stamp = self.make_stamp()
        qx, qy, qz, qw = quaternion_from_yaw(float(x[2]))

        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = self.frame_id
        out.child_frame_id = self.child_frame_id
        out.pose.pose.position.x = float(x[0])
        out.pose.pose.position.y = float(x[1])
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.pose.covariance = cov6_from_cov3(P[:3, :3])
        # twist is expressed in child_frame_id (robot frame): vx = v, wz = omega
        out.twist.twist.linear.x = float(x[3])
        out.twist.twist.angular.z = float(x[4])
        out.twist.covariance = twist_cov6(P[3, 3], P[4, 4], P[3, 4])
        self.odom_pub.publish(out)

        t = self.fusion.time
        if self.last_path_time is None or t - self.last_path_time >= self.path_period:
            self.last_path_time = t
            pose = PoseStamped()
            pose.header = out.header
            pose.pose = out.pose.pose
            self.path.header.stamp = stamp
            self.path.poses.append(pose)
            self.path_pub.publish(self.path)

    # ------------------------------------------------------------- evaluation
    def save_evaluation_csv(self):
        """Write the pre-update errors at the GT instants (plot_results.py format)."""
        log = self.fusion.gt_log
        if not log:
            self.get_logger().warn('No ground-truth messages received; no CSV written.')
            return
        directory = os.path.dirname(self.eval_csv)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.eval_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(r.as_row() for r in log)
        self.get_logger().info(f'Evaluation data ({len(log)} rows) saved to {self.eval_csv}')


def main(args=None):
    """Run the EKF node until Ctrl+C, then write the evaluation CSV."""
    rclpy.init(args=args)
    node = EkfNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.save_evaluation_csv()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
