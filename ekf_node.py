#!/usr/bin/env python3
"""
ekf_node.py - EKF proprio para a Task 1a (Mini-Projeto 1, IRobo 2026/2027)

Estado estimado:  s = [x, y, theta]
    Pose do base_footprint no referencial 'odom'.

PREDICT  (a cada mensagem da IMU, ~116 Hz):
    v     = velocidade linear mais recente de /odom (~25 Hz)
    omega = velocidade angular z da IMU
    Modelo de movimento (robot diferencial):
        x     <- x + v*dt*cos(theta)
        y     <- y + v*dt*sin(theta)
        theta <- theta + omega*dt

UPDATE   (a 1 Hz, com o ground truth):
    O ground truth e a posicao do LASER (TF odom -> mocap_laser_link),
    nao a do base_footprint. Por isso o modelo de medicao inclui o
    offset (ox, oy) do laser em relacao ao base_footprint:
        h(s) = [x + ox*cos(theta) - oy*sin(theta),
                y + ox*sin(theta) + oy*cos(theta)]

PUBLICA:
    /ekf/odom     nav_msgs/Odometry  pose + covariancia (display Odometry no RViz)
    /ekf/path     nav_msgs/Path      trajetoria estimada do laser
    /ekf/gt_path  nav_msgs/Path      trajetoria do ground truth (so visualizacao)
    TF odom -> ekf_footprint -> ekf_scan   (para o calculate_error.py)

Uso:
    ros2 run ekf_localization ekf_node --ros-args -p use_sim_time:=true
    # Parametros ajustaveis (exemplo):
    ros2 run ekf_localization ekf_node --ros-args -p use_sim_time:=true -p q_w:=0.05 -p r_gt:=0.02
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import (Buffer, TransformListener, TransformBroadcaster,
                     StaticTransformBroadcaster, LookupException,
                     ConnectivityException, ExtrapolationException)

TF_ERRORS = (LookupException, ConnectivityException, ExtrapolationException)


def wrap_angle(a):
    """Normaliza um angulo para [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class EkfNode(Node):

    def __init__(self):
        super().__init__('ekf_node')

        # ---------------- Parametros do filtro ----------------
        # q_v, q_w: intensidade do ruido de processo (variancia por segundo)
        #   q_v em (m/s)^2*s, q_w em (rad/s)^2*s
        # r_gt: desvio-padrao da medicao do ground truth [m]
        # gt_rate: frequencia do update com ground truth [Hz]
        self.q_v = self.declare_parameter('q_v', 0.01).value
        self.q_w = self.declare_parameter('q_w', 0.01).value
        self.r_gt = self.declare_parameter('r_gt', 0.01).value
        self.gt_rate = self.declare_parameter('gt_rate', 1.0).value

        # ---------------- Estado do filtro ----------------
        self.s = np.zeros(3)        # [x, y, theta]
        self.P = np.eye(3)          # covariancia
        self.initialized = False
        self.last_t = None          # tempo (s) da ultima predicao
        self.v = None               # ultima velocidade linear da odometria
        self.odom_yaw = None        # ultima orientacao da odometria (so para inicializar)
        self.offset = None          # (ox, oy): laser no referencial base_footprint
        self.n_predict = 0

        # ---------------- TF ----------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # ---------------- Subscricoes ----------------
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu', self.imu_cb, qos_profile_sensor_data)

        # ---------------- Publicadores ----------------
        self.odom_pub = self.create_publisher(Odometry, '/ekf/odom', 10)
        self.path_pub = self.create_publisher(Path, '/ekf/path', 10)
        self.gt_path_pub = self.create_publisher(Path, '/ekf/gt_path', 10)
        self.path = Path()
        self.path.header.frame_id = 'odom'
        self.gt_path = Path()
        self.gt_path.header.frame_id = 'odom'

        # ---------------- Temporizadores (tempo simulado) ----------------
        self.create_timer(1.0 / self.gt_rate, self.gt_update_cb)   # update a 1 Hz
        self.create_timer(0.1, self.gt_path_cb)                    # GT path a 10 Hz (so visualizacao)

        self.get_logger().info(
            f'EKF iniciado: q_v={self.q_v}, q_w={self.q_w}, '
            f'r_gt={self.r_gt} m, update GT a {self.gt_rate} Hz')

    # ==================================================================
    # Callbacks dos sensores
    # ==================================================================
    def odom_cb(self, msg):
        # Da odometria so usamos a velocidade linear (e a orientacao para inicializar).
        self.v = msg.twist.twist.linear.x
        self.odom_yaw = yaw_from_quat(msg.pose.pose.orientation)

    def imu_cb(self, msg):
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

        # Bag reiniciado: o tempo simulado saltou para tras
        if self.last_t is not None and t < self.last_t - 0.5:
            self.get_logger().warn('Salto para tras no tempo - a reiniciar o filtro.')
            self.reset()

        if not self.initialized or self.v is None:
            self.last_t = t
            return

        dt = t - self.last_t
        self.last_t = t
        if dt <= 0.0 or dt > 0.5:
            return

        self.predict(self.v, msg.angular_velocity.z, dt)
        self.publish(msg.header.stamp)

    # ==================================================================
    # EKF: predict
    # ==================================================================
    def predict(self, v, w, dt):
        x, y, th = self.s
        c, s = math.cos(th), math.sin(th)

        # Estado previsto
        self.s = np.array([x + v * dt * c,
                           y + v * dt * s,
                           wrap_angle(th + w * dt)])

        # Jacobiano do modelo de movimento em ordem ao estado
        F = np.array([[1.0, 0.0, -v * dt * s],
                      [0.0, 1.0,  v * dt * c],
                      [0.0, 0.0,  1.0]])

        # Ruido de processo: ruido em (v, omega) projetado no estado.
        # Multiplicar por dt torna Q independente da frequencia da IMU.
        B = np.array([[c, 0.0],
                      [s, 0.0],
                      [0.0, 1.0]])
        Q = B @ np.diag([self.q_v, self.q_w]) @ B.T * dt

        self.P = F @ self.P @ F.T + Q

    # ==================================================================
    # EKF: update com o ground truth (1 Hz)
    # ==================================================================
    def gt_update_cb(self):
        if not self.get_offset():
            return
        gt = self.lookup_gt()
        if gt is None:
            return
        zx, zy, _ = gt

        if not self.initialized:
            self.initialize(zx, zy)
            return

        x, y, th = self.s
        ox, oy = self.offset
        c, s = math.cos(th), math.sin(th)

        # Medicao prevista: onde o laser deveria estar segundo o estado atual
        h = np.array([x + ox * c - oy * s,
                      y + ox * s + oy * c])

        # Jacobiano da medicao em ordem ao estado
        H = np.array([[1.0, 0.0, -ox * s - oy * c],
                      [0.0, 1.0,  ox * c - oy * s]])

        R = np.eye(2) * self.r_gt ** 2
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)       # ganho de Kalman

        innov = np.array([zx, zy]) - h            # inovacao
        self.s = self.s + K @ innov
        self.s[2] = wrap_angle(self.s[2])

        # Forma de Joseph: numericamente mais estavel que (I - KH) P
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self.get_logger().info(
            f'Update GT | inovacao = {np.linalg.norm(innov) * 1000:6.1f} mm | '
            f'sigma_xy = {math.sqrt(self.P[0, 0]) * 1000:5.1f} mm | '
            f'sigma_theta = {math.degrees(math.sqrt(self.P[2, 2])):4.2f} deg')

    def initialize(self, zx, zy):
        """Inicializa o estado com o primeiro ground truth e a orientacao da odometria."""
        if self.odom_yaw is None:
            return
        th = self.odom_yaw
        ox, oy = self.offset
        c, s = math.cos(th), math.sin(th)
        self.s = np.array([zx - (ox * c - oy * s),
                           zy - (ox * s + oy * c),
                           th])
        self.P = np.diag([self.r_gt ** 2, self.r_gt ** 2, math.radians(5.0) ** 2])
        self.initialized = True
        self.get_logger().info(
            f'Filtro inicializado em x={self.s[0]:.3f}, y={self.s[1]:.3f}, '
            f'theta={math.degrees(th):.1f} deg')

    def reset(self):
        self.initialized = False
        self.last_t = None
        self.n_predict = 0
        self.path.poses = []
        self.gt_path.poses = []

    # ==================================================================
    # TF auxiliares
    # ==================================================================
    def get_offset(self):
        """Le (uma vez) a posicao do laser (base_scan) no base_footprint."""
        if self.offset is not None:
            return True
        try:
            tf = self.tf_buffer.lookup_transform('base_footprint', 'base_scan', Time())
        except TF_ERRORS:
            return False
        tr = tf.transform.translation
        self.offset = (tr.x, tr.y)

        # TF estatico ekf_footprint -> ekf_scan (mesmo offset do robot real)
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = 'ekf_footprint'
        ts.child_frame_id = 'ekf_scan'
        ts.transform.translation.x = tr.x
        ts.transform.translation.y = tr.y
        ts.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(ts)

        self.get_logger().info(f'Offset do laser: ox={tr.x:.3f} m, oy={tr.y:.3f} m')
        return True

    def lookup_gt(self):
        """Posicao do ground truth (laser) no referencial odom."""
        try:
            tf = self.tf_buffer.lookup_transform('odom', 'mocap_laser_link', Time())
        except TF_ERRORS as e:
            self.get_logger().warn(f'Ground truth indisponivel: {e}',
                                   throttle_duration_sec=2.0)
            return None
        tr = tf.transform.translation
        return tr.x, tr.y, tf.header.stamp

    # ==================================================================
    # Publicacao
    # ==================================================================
    def publish(self, stamp):
        x, y, th = self.s
        qz, qw = math.sin(th / 2.0), math.cos(th / 2.0)

        # Odometry com covariancia (6x6: x, y, z, roll, pitch, yaw)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'ekf_footprint'
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        cov = [0.0] * 36
        idx = [0, 1, 5]  # x, y, yaw na matriz 6x6
        for i in range(3):
            for j in range(3):
                cov[idx[i] * 6 + idx[j]] = float(self.P[i, j])
        cov[14] = cov[21] = cov[28] = 1e-9   # z, roll, pitch (2D)
        odom.pose.covariance = cov
        self.odom_pub.publish(odom)

        # TF odom -> ekf_footprint
        ts = TransformStamped()
        ts.header.stamp = stamp
        ts.header.frame_id = 'odom'
        ts.child_frame_id = 'ekf_footprint'
        ts.transform.translation.x = x
        ts.transform.translation.y = y
        ts.transform.rotation.z = qz
        ts.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(ts)

        # Path do laser estimado (1 em cada 10 predicoes, ~12 Hz),
        # o mesmo ponto que o ground truth segue.
        self.n_predict += 1
        if self.n_predict % 10 == 0 and self.offset is not None:
            ox, oy = self.offset
            c, s = math.cos(th), math.sin(th)
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = 'odom'
            ps.pose.position.x = x + ox * c - oy * s
            ps.pose.position.y = y + ox * s + oy * c
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            self.path.poses.append(ps)
            self.path.header.stamp = stamp
            self.path_pub.publish(self.path)

    def gt_path_cb(self):
        """Trajetoria do ground truth a 10 Hz - so para visualizacao, nao entra no filtro."""
        gt = self.lookup_gt()
        if gt is None:
            return
        zx, zy, stamp = gt
        if self.gt_path.poses and self.gt_path.poses[-1].header.stamp == stamp:
            return
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = 'odom'
        ps.pose.position.x = zx
        ps.pose.position.y = zy
        ps.pose.orientation.w = 1.0
        self.gt_path.poses.append(ps)
        self.gt_path.header.stamp = stamp
        self.gt_path_pub.publish(self.gt_path)


def main(args=None):
    rclpy.init(args=args)
    node = EkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
