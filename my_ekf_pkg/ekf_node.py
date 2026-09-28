import csv
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import numpy as np
from my_ekf_pkg.ekf import EKF

class EKFNode(Node):
    def __init__(self):
        super().__init__("ekf_node")

        #Initialise state vector [x, y, theta] and covariance matrix P
        self.x = np.zeros((3))
        self.P = np.eye(3) * 0.1

        #Noise matrices for process and measurement noise
        #The numbers are placeholders and need to be tuned
        self.Q = np.diag([0.02, 0.02, 0.01])
        self.R = np.diag([0.001, 0.001, 0.001])

        self.ekf = EKF(self.x, self.P, self.Q, self.R)

        #Store last received velocity and angular velocity
        self.last_v = 0.0
        self.last_omega = 0.0

        #Store last time for prediction and last time for ground truth update
        self.last_time = None
        self.last_gt_time = None
        #Minimum time between ground truth updates in seconds (1.0 sec = 1 Hz)
        self.gt_period = 1.0

        #Store evaluation data
        self.eval_data = []

        #Create subscriptions for odometry, IMU, and ground truth data
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/ground_truth', self.gt_callback, 10)

        #Create publisher for estimated state
        self.ekf_pub = self.create_publisher(Odometry, '/ekf/odom', 10)

        #Log that the EKF node started
        self.get_logger().info("EKF Node has been started.")

    #Combine sec and nanosec fields of ROS timestamps into one float (in sec)
    def stamp_to_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    #Run the prediction step of the EKF using the last received velocity and angular velocity
    def run_predict(self, stamp):
        t = self.stamp_to_sec(stamp)

        if self.last_time is None:
            self.last_time = t
            return

        dt = t - self.last_time
        if dt <= 0.0 or dt > 1.0:
            self.last_time = t
            return

        self.ekf.predict(self.last_v, self.last_omega, dt)
        self.last_time = t

        self.publish_estimate(stamp)

    #Callback for odometry messages, updates the last received linear velocity and runs the prediction step
    def odom_callback(self, msg: Odometry):
        self.last_v = msg.twist.twist.linear.x
        self.run_predict(msg.header.stamp)

    #Callback for IMU messages, updates the last received angular velocity and runs the prediction step
    def imu_callback(self, msg: Imu):
        self.last_omega = msg.angular_velocity.z
        self.run_predict(msg.header.stamp)

    #Callback for ground truth messages, updates the EKF with the ground truth position and orientation if enough time has passed since the last update
    def gt_callback(self, msg: Odometry):
        t = self.stamp_to_sec(msg.header.stamp)

        #Downsample ground truth to 1 Hz
        if self.last_gt_time is not None and (t - self.last_gt_time) < self.gt_period:
            return

        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        z = np.array([msg.pose.pose.position.x,
                      msg.pose.pose.position.y,
                      yaw])

        #Error calculation before measurement update
        err_x = self.ekf.x[0] - z[0]
        err_y = self.ekf.x[1] - z[1]
        err_dist = np.sqrt(err_x**2 + err_y**2)
        err_theta = self.ekf.normalize_angle(self.ekf.x[2] - z[2])

        self.eval_data.append([t, err_dist, err_x, err_y, err_theta])
        
        self.ekf.update(z)
        self.last_gt_time = t
        self.publish_estimate(msg.header.stamp)

    #Publish the current state estimate as an Odometry message
    def publish_estimate(self, stamp):
        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = "map"
        out.pose.pose.position.x = float(self.ekf.x[0])
        out.pose.pose.position.y = float(self.ekf.x[1])

        theta = float(self.ekf.x[2])
        out.pose.pose.orientation.z = np.sin(theta / 2.0)
        out.pose.pose.orientation.w = np.cos(theta / 2.0)

        self.ekf_pub.publish(out)

    def save_evaluation_csv(self):
        if not self.eval_data:
            return
        
        file_path = os.path.expanduser("~/ros2_ws/ekf_evaluation.csv")
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'error_dist', 'err_x', 'err_y', 'err_yaw'])
            writer.writerows(self.eval_data)
        self.get_logger().info(f"Evaluation data saved to {file_path}")


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_evaluation_csv()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
