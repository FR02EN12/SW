from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _arg(name: str) -> LaunchConfiguration:
    return LaunchConfiguration(name)


def _run(package: str, executable: str, name: str, parameters=None):
    return Node(
        package=package,
        executable=executable,
        name=name,
        output='screen',
        condition=IfCondition(_arg('use_sw_tail')),
        parameters=parameters or [],
    )


def generate_launch_description():
    image_topic = _arg('image_topic')
    camera_info_topic = _arg('camera_info_topic')
    perf_cfg = PathJoinSubstitution([
        FindPackageShare('bringup'),
        'launch',
        'sw_params.yaml',
    ])
    lane_cam_cfg = PathJoinSubstitution([
        FindPackageShare('perception_py'),
        'config',
        'fisheye.yaml',
    ])
    tb3_param_dir = PathJoinSubstitution([
        FindPackageShare('turtlebot3_bringup'),
        'param',
        'humble',
        'burger.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument('use_base', default_value='true'),
        DeclareLaunchArgument('use_lidar', default_value='true'),
        DeclareLaunchArgument('use_camera', default_value='true'),
        DeclareLaunchArgument('use_lane_frontend', default_value='true'),
        DeclareLaunchArgument('use_sw_tail', default_value='true'),
        DeclareLaunchArgument('opencr_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('lidar_port', default_value='/dev/tb3_lidar'),
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        DeclareLaunchArgument('image_topic', default_value='/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera_info'),
        DeclareLaunchArgument('image_width', default_value='640'),
        DeclareLaunchArgument('image_height', default_value='480'),
        DeclareLaunchArgument('pixel_format', default_value='yuyv2rgb'),
        DeclareLaunchArgument('framerate', default_value='30.0'),
        DeclareLaunchArgument('camera_name', default_value='fisheye'),
        DeclareLaunchArgument('lane_show', default_value='false'),
        DeclareLaunchArgument('led_backend', default_value='arduino_serial'),
        DeclareLaunchArgument('arduino_port', default_value='/dev/ttyACM1'),
        DeclareLaunchArgument('arduino_baudrate', default_value='9600'),

        Node(
            package='coin_d4_driver',
            executable='single_coin_d4_node',
            name='single_coin_d4_node',
            output='screen',
            condition=IfCondition(_arg('use_lidar')),
            parameters=[{
                'port': _arg('lidar_port'),
                'frame_id': 'base_scan',
                'baudrate': 230400,
                'version': 4,
                'topic_name': 'scan',
                'reverse': False,
                'warmup_time': 5,
            }],
        ),

        Node(
            package='turtlebot3_node',
            executable='turtlebot3_ros',
            name='turtlebot3_ros',
            output='screen',
            condition=IfCondition(_arg('use_base')),
            parameters=[tb3_param_dir, {'namespace': ''}],
            arguments=['-i', _arg('opencr_port')],
        ),

        Node(
            package='perception_py',
            executable='camera_driver',
            name='camera_driver',
            output='screen',
            condition=IfCondition(_arg('use_camera')),
            parameters=[{
                'video_device': _arg('video_device'),
                'image_width': ParameterValue(_arg('image_width'), value_type=int),
                'image_height': ParameterValue(_arg('image_height'), value_type=int),
                'framerate': ParameterValue(_arg('framerate'), value_type=float),
                'pixel_format': _arg('pixel_format'),
                'camera_name': _arg('camera_name'),
                'image_topic': image_topic,
                'camera_info_topic': camera_info_topic,
            }],
        ),

        Node(
            package='perception_py',
            executable='lane_final',
            name='lane_final',
            output='screen',
            condition=IfCondition(_arg('use_lane_frontend')),
            parameters=[{
                'image_topic': image_topic,
                'camera_info_topic': camera_info_topic,
                'camera_info_yaml': lane_cam_cfg,
                'use_camera_rectification': True,
                'show': ParameterValue(_arg('lane_show'), value_type=bool),
                'show_debug_view': ParameterValue(_arg('lane_show'), value_type=bool),
            }],
        ),

        # SW perception/decision/control tail. The external base bringup owns
        # robot hardware topics such as /scan, /odom, /imu, and /cmd_vel.
        _run('perception_cpp', 'localization_slam', 'localization_slam'),
        _run(
            'perception_py',
            'led_signal',
            'led_signal',
            [perf_cfg, {
                'image_topic': image_topic,
                'camera_info_yaml': lane_cam_cfg,
            }],
        ),
        _run('perception_py', 'space_memory', 'space_memory'),
        _run('perception_py', 'situation_fusion', 'situation_fusion', [perf_cfg]),

        _run('decision_cpp', 'path_plan', 'path_plan'),
        _run('decision_cpp', 'safety_supervisor', 'safety_supervisor'),
        _run(
            'decision_py',
            'drive_mode',
            'drive_mode',
            [perf_cfg],
        ),
        _run(
            'decision_py',
            'cooperation',
            'cooperation',
            [perf_cfg],
        ),
        _run(
            'decision_py',
            'behavior_plan',
            'behavior_plan',
            [perf_cfg],
        ),

        _run(
            'control_py',
            'lane_follow',
            'lane_follow',
            [perf_cfg],
        ),
        _run(
            'control_cpp',
            'path_track',
            'path_track',
            [perf_cfg],
        ),
        _run('control_cpp', 'cmd_mux', 'cmd_mux'),
        _run('control_py', 'led_behavior', 'led_behavior', [perf_cfg]),
        _run(
            'control_py',
            'led_io',
            'led_io',
            [perf_cfg, {
                'backend': _arg('led_backend'),
                'arduino_port': _arg('arduino_port'),
                'arduino_baudrate': ParameterValue(
                    _arg('arduino_baudrate'), value_type=int),
            }],
        ),
    ])
