"""
Publish the motion-capture ground truth as pose topics in frame ``odom``.

The dataset has no ground-truth topic: the ground truth is the TF
``mocap -> mocap_laser_link`` (60 Hz) inside /tf, and it tracks the LASER,
not the robot centre. This node converts every sample to the pose of
``base_footprint`` in ``odom`` and publishes:

* /gt/pose_full  (geometry_msgs/PoseStamped, every sample) - for evaluation
* /gt/pose_1hz   (geometry_msgs/PoseWithCovarianceStamped, 1 Hz) - EKF input
* /gt/path       (nav_msgs/Path, ~10 Hz) - for RViz
"""

import math

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from my_ekf_pkg.geometry import quaternion_from_yaw
from my_ekf_pkg.ground_truth import GroundTruthConverter, RateLimiter
from my_ekf_pkg.msg_utils import cov6_from_cov3, pose2d_from_transform, stamp_to_sec
from nav_msgs.msg import Path
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

PARAMS = {
    'mocap_frame': 'mocap',
    'gt_frame': 'mocap_laser_link',
    'target_frame': 'odom',
    'sensor_offset_x': -0.064,
    'sensor_offset_y': 0.0,
    'rate_hz': 1.0,
    'sigma_xy': 0.01,
    'sigma_yaw': 0.0175,
    'path_period': 0.1,
    'time_jump_reset': 1.0,
    'tf_topic': '/tf',
    'full_topic': '/gt/pose_full',
    'rate_topic': '/gt/pose_1hz',
    'path_topic': '/gt/path',
}

# The first GT pose should be at the odom origin; warn if it is further away
START_TOL_XY = 0.05
START_TOL_YAW = math.radians(5.0)


class GtPublisher(Node):
    """Convert mocap TF samples into ground-truth pose topics."""

    def __init__(self):
        """Declare parameters and set up TF, publishers and the /tf subscriber."""
        super().__init__('gt_publisher')
        any_type = ParameterDescriptor(dynamic_typing=True)
        for name, value in PARAMS.items():
            self.declare_parameter(name, value, any_type)
        p = {name: self.get_parameter(name).value for name in PARAMS}

        self.mocap_frame = str(p['mocap_frame'])
        self.gt_frame = str(p['gt_frame'])
        self.target_frame = str(p['target_frame'])
        self.path_period = float(p['path_period'])
        self.time_jump_reset = float(p['time_jump_reset'])
        sxy, syaw = float(p['sigma_xy']), float(p['sigma_yaw'])
        self.covariance = cov6_from_cov3([[sxy ** 2, 0.0, 0.0],
                                          [0.0, sxy ** 2, 0.0],
                                          [0.0, 0.0, syaw ** 2]])

        self.converter = GroundTruthConverter(
            sensor_offset=(float(p['sensor_offset_x']), float(p['sensor_offset_y'])))
        self.limiter = RateLimiter(float(p['rate_hz']), self.time_jump_reset)

        # Only used to look up the static odom <- mocap transform once
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.full_pub = self.create_publisher(PoseStamped, str(p['full_topic']), 100)
        self.rate_pub = self.create_publisher(PoseWithCovarianceStamped,
                                              str(p['rate_topic']), 10)
        self.path_pub = self.create_publisher(Path, str(p['path_topic']), 10)

        tf_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                            reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(TFMessage, str(p['tf_topic']), self.tf_callback, tf_qos)

        self.path = Path()
        self.path.header.frame_id = self.target_frame
        self.last_path_time = None
        self.last_time = None
        self.first_logged = False
        self.get_logger().info(
            f'Publishing ground truth from TF {self.mocap_frame} -> {self.gt_frame} '
            f'in frame {self.target_frame}')

    def reset_outputs(self):
        """Restart downsampling and the path (bag restarted)."""
        self.limiter.reset()
        self.path.poses = []
        self.last_path_time = None
        self.first_logged = False

    def ensure_static_transform(self):
        """Look up T_odom_mocap once; return True when it is available."""
        if self.converter.ready:
            return True
        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, self.mocap_frame, Time())
        except TransformException as ex:
            self.get_logger().warn(
                f'No transform {self.target_frame} <- {self.mocap_frame} yet '
                f'(is publish_initial_tf running?): {ex}', throttle_duration_sec=5.0)
            return False
        self.converter.set_odom_mocap(pose2d_from_transform(tf.transform))
        x, y, yaw = self.converter.T_odom_mocap
        self.get_logger().info(
            f'T_{self.target_frame}_{self.mocap_frame} = '
            f'({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f} deg)')
        return True

    def tf_callback(self, msg):
        """Handle the ground-truth transforms contained in a /tf message."""
        for tr in msg.transforms:
            if (tr.header.frame_id.lstrip('/') == self.mocap_frame
                    and tr.child_frame_id.lstrip('/') == self.gt_frame):
                self.handle_sample(tr)

    def handle_sample(self, tr):
        """Convert one mocap sample and publish it."""
        t = stamp_to_sec(tr.header.stamp)
        if self.last_time is not None and t < self.last_time - self.time_jump_reset:
            self.get_logger().warn('Time jumped back (bag restarted?): resetting GT outputs.')
            self.reset_outputs()
        self.last_time = t
        if not self.ensure_static_transform():
            return

        x, y, yaw = self.converter.convert(pose2d_from_transform(tr.transform))
        if not self.first_logged:
            self.first_logged = True
            self.log_first_pose(x, y, yaw)

        pose = PoseStamped()
        pose.header.stamp = tr.header.stamp
        pose.header.frame_id = self.target_frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = quaternion_from_yaw(yaw)
        self.full_pub.publish(pose)

        if self.limiter.accept(t):
            out = PoseWithCovarianceStamped()
            out.header = pose.header
            out.pose.pose = pose.pose
            out.pose.covariance = self.covariance
            self.rate_pub.publish(out)

        if self.last_path_time is None or t - self.last_path_time >= self.path_period:
            self.last_path_time = t
            self.path.header.stamp = tr.header.stamp
            self.path.poses.append(pose)
            self.path_pub.publish(self.path)

    def log_first_pose(self, x, y, yaw):
        """Log the first GT pose and warn if it is not at the odom origin."""
        text = (f'First GT pose in {self.target_frame}: '
                f'x={x:.3f} m, y={y:.3f} m, yaw={math.degrees(yaw):.2f} deg')
        if math.hypot(x, y) > START_TOL_XY or abs(yaw) > START_TOL_YAW:
            self.get_logger().warn(
                text + ' - expected ~(0, 0, 0). Check sensor_offset_x/y and the '
                'static mocap -> odom transform.')
        else:
            self.get_logger().info(text)


def main(args=None):
    """Run the ground-truth publisher."""
    rclpy.init(args=args)
    node = GtPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
