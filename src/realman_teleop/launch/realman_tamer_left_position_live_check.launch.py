from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    live_confirmation = LaunchConfiguration("live_confirmation")

    return LaunchDescription(
        [
            DeclareLaunchArgument("live_confirmation", default_value=""),
            SetEnvironmentVariable(
                "RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] {message}"
            ),
            Node(
                package="vr_data_pub",
                executable="vr_data_pub",
                name="vr_socket_server",
                parameters=[{"server_port": 8018}],
                output="log",
            ),
            Node(
                package="vr_data_pub",
                executable="vr_data_distributer",
                name="vr_data_distributor",
                output="log",
            ),
            Node(
                package="realman_teleop",
                namespace="left_arm",
                executable="realman_pose_converter",
                name="vr_robot_pose_converter",
                parameters=[
                    {
                        "robot_name": "left",
                        "position_axis_map": "z,-y,-x",
                        "rotation_axis_map": "-z,y,-x",
                        "translation_reference_frame": "controller_anchor_yaw",
                        "controller_forward_axis": "z",
                    }
                ],
                remappings=[
                    ("robot_command_servo_p", "interpolated_robot_command")
                ],
                output="log",
            ),
            Node(
                package="realman_teleop",
                executable="realman_dual_arm_bridge",
                name="realman_left_arm_bridge",
                parameters=[
                    {
                        "managed_arms": "left",
                        "left_ip": "169.254.128.18",
                        "sdk_thread_mode": "dual",
                        "cpu_affinity": "4,5",
                        "dry_run": False,
                        "live_confirmation": live_confirmation,
                        "command_rate_hz": 125.0,
                        "state_rate_hz": 2.0,
                        "follow": True,
                        "trajectory_mode": 0,
                        "radio": 0,
                        "canfd_pose_format": "quaternion",
                        "max_send_interval_seconds": 0.020,
                        "hard_send_interval_seconds": 0.050,
                        "max_consecutive_send_gap_violations": 3,
                        "independent_canfd_thread": True,
                        "direct_target_passthrough": False,
                        "pose_step_limit_enabled": False,
                        "pose_acceleration_limit_enabled": True,
                        "position_mapping_scale": 0.40,
                        "rotation_mapping_scale": 0.05,
                        "max_linear_velocity_mps": 0.15,
                        "max_linear_acceleration_mps2": 1.2,
                        "max_angular_velocity_radps": 0.5,
                        "max_angular_acceleration_radps2": 4.0,
                        "watchdog_seconds": 0.15,
                        "trigger_control_enabled": True,
                        "trigger_mode": "hold",
                        "slow_stop_on_trigger_release": True,
                        "left_control_enabled": True,
                        "right_control_enabled": False,
                        "orientation_control_enabled": True,
                        "workspace_limit_m": 0.04,
                        "rotation_workspace_limit_rad": 0.25,
                        "initial_position_tolerance_m": 0.01,
                        "initial_rotation_tolerance_rad": 0.10,
                        "emergency_stop_service": "/left_arm/emergency_stop",
                    }
                ],
                output="screen",
            ),
        ]
    )
