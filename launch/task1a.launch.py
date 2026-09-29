"""
Task 1a: play the dataset and run our EKF plus robot_localization.

Examples::

    ros2 launch my_ekf_pkg task1a.launch.py
    ros2 launch my_ekf_pkg task1a.launch.py record:=true bag_out:=task1a_results
    ros2 launch my_ekf_pkg task1a.launch.py use_rl:=false viz:=foxglove
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

RECORD_TOPICS = [
    '/ekf/odom', '/ekf/path', '/rl/no_gt/odom', '/rl/gt/odom',
    '/gt/pose_full', '/gt/pose_1hz', '/gt/path', '/odom',
]


def generate_launch_description():
    """Build the launch description."""
    play_bag = LaunchConfiguration('play_bag')
    viz = LaunchConfiguration('viz')
    use_rl = LaunchConfiguration('use_rl')
    record = LaunchConfiguration('record')
    bag_out = LaunchConfiguration('bag_out')
    params_file = LaunchConfiguration('params_file')
    rl_params_file = LaunchConfiguration('rl_params_file')

    args = [
        DeclareLaunchArgument('play_bag', default_value='true',
                              description='Play the dataset (turtlebot_playbag.launch.py)'),
        DeclareLaunchArgument('viz', default_value='rviz2', choices=['rviz2', 'foxglove'],
                              description='rviz2: the RViz started by the dataset launch; '
                                          'foxglove: additionally start foxglove_bridge'),
        DeclareLaunchArgument('use_rl', default_value='true',
                              description='Also run the two robot_localization EKFs'),
        DeclareLaunchArgument('record', default_value='false',
                              description='Record the results with ros2 bag record'),
        DeclareLaunchArgument('bag_out', default_value='task1a_results',
                              description='Output directory of the recorded bag'),
        DeclareLaunchArgument('params_file',
                              default_value=PathJoinSubstitution(
                                  [FindPackageShare('my_ekf_pkg'), 'config',
                                   'ekf_params.yaml']),
                              description='Parameters of ekf_node and gt_publisher'),
        DeclareLaunchArgument('rl_params_file',
                              default_value=PathJoinSubstitution(
                                  [FindPackageShare('my_ekf_pkg'), 'config', 'ekf_rl.yaml']),
                              description='Parameters of the robot_localization EKFs'),
    ]

    # Dataset playback (bag with --clock, robot_state_publisher, RViz, and the
    # static mocap -> odom transform from publish_initial_tf)
    playbag = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('turtlebot_datasets'), 'launch',
             'turtlebot_playbag.launch.py'])),
        launch_arguments={'fixed_frame': 'odom'}.items(),
        condition=IfCondition(play_bag))

    sim_time = {'use_sim_time': True}
    gt_publisher = Node(
        package='my_ekf_pkg', executable='gt_publisher', name='gt_publisher',
        output='screen', parameters=[params_file, sim_time])
    ekf_node = Node(
        package='my_ekf_pkg', executable='ekf_node', name='ekf_node',
        output='screen', parameters=[params_file, sim_time])

    rl_nodes = [
        Node(package='robot_localization', executable='ekf_node', name=name,
             output='screen', parameters=[rl_params_file, sim_time],
             remappings=[('odometry/filtered', topic)],
             condition=IfCondition(use_rl))
        for name, topic in (('ekf_rl_no_gt', '/rl/no_gt/odom'),
                            ('ekf_rl_gt', '/rl/gt/odom'))
    ]

    recorder = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', bag_out] + RECORD_TOPICS,
        output='screen', condition=IfCondition(record))

    foxglove = Node(
        package='foxglove_bridge', executable='foxglove_bridge', name='foxglove_bridge',
        output='screen', parameters=[sim_time],
        condition=IfCondition(PythonExpression(["'", viz, "' == 'foxglove'"])))

    return LaunchDescription(
        args + [playbag, gt_publisher, ekf_node] + rl_nodes + [recorder, foxglove])
