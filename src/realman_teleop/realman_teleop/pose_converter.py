import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from std_srvs.srv import SetBool


def parse_axis_map(value: str) -> np.ndarray:
    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    names = [item.strip().lower() for item in value.split(",")]
    if len(names) != 3 or any(name not in axes for name in names):
        raise ValueError("axis_map must contain three axes, for example 'z,y,-x'")
    matrix = np.vstack([axes[name] for name in names])
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-8):
        raise ValueError("axis_map axes must be orthogonal and unique")
    if not math.isclose(abs(float(np.linalg.det(matrix))), 1.0, abs_tol=1e-8):
        raise ValueError("axis_map must be an orthogonal coordinate transform")
    return matrix


def pico_euler_to_rotation(euler_degrees) -> Rotation:
    rx, ry, rz = np.radians([float(value) for value in euler_degrees])
    rotation_x = np.array(
        [[1, 0, 0], [0, np.cos(-rx), -np.sin(-rx)], [0, np.sin(-rx), np.cos(-rx)]]
    )
    rotation_y = np.array(
        [[np.cos(-ry), 0, np.sin(-ry)], [0, 1, 0], [-np.sin(-ry), 0, np.cos(-ry)]]
    )
    rotation_z = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )
    return Rotation.from_matrix(rotation_y @ rotation_x @ rotation_z)


def position_delta_in_reference_frame(
    controller_delta,
    controller_anchor_rotation: Rotation,
    reference_frame: str,
    controller_forward_axis: str = "z",
) -> np.ndarray:
    delta = np.asarray(controller_delta, dtype=float)
    if reference_frame == "tracking":
        return delta
    if reference_frame == "controller_anchor_full":
        return controller_anchor_rotation.inv().apply(delta)
    if reference_frame == "controller_anchor_yaw":
        forward_axes = {
            "x": np.array([1.0, 0.0, 0.0]),
            "-x": np.array([-1.0, 0.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
            "-z": np.array([0.0, 0.0, -1.0]),
        }
        if controller_forward_axis not in forward_axes:
            raise ValueError("controller_forward_axis must be x, -x, z, or -z")
        # PICO Euler yaw is converted with the opposite sign from tracking-space
        # position yaw, so recover the physical controller heading with the inverse.
        forward = controller_anchor_rotation.inv().apply(
            forward_axes[controller_forward_axis]
        )
        horizontal_forward = np.array([forward[0], 0.0, forward[2]], dtype=float)
        norm = float(np.linalg.norm(horizontal_forward))
        if norm < 1e-6:
            raise ValueError("controller forward axis is vertical; cannot determine anchor yaw")
        horizontal_forward /= norm
        yaw = math.atan2(horizontal_forward[0], horizontal_forward[2])
        return Rotation.from_euler("y", yaw).inv().apply(delta)
    raise ValueError(
        "translation_reference_frame must be tracking, controller_anchor_yaw, "
        "or controller_anchor_full"
    )


def map_relative_pose(
    robot_anchor_position,
    robot_anchor_rotation: Rotation,
    controller_anchor_position,
    controller_anchor_rotation: Rotation,
    controller_position,
    controller_rotation: Rotation,
    position_axis_matrix: np.ndarray,
    rotation_axis_matrix: np.ndarray,
    translation_reference_frame: str = "tracking",
    controller_forward_axis: str = "-z",
):
    controller_delta = np.asarray(controller_position, dtype=float) - np.asarray(
        controller_anchor_position, dtype=float
    )
    controller_delta = position_delta_in_reference_frame(
        controller_delta,
        controller_anchor_rotation,
        translation_reference_frame,
        controller_forward_axis,
    )
    target_position = (
        np.asarray(robot_anchor_position, dtype=float)
        + position_axis_matrix @ controller_delta
    )
    controller_relative_rotation = controller_anchor_rotation.inv() * controller_rotation
    mapped_rotation_vector = rotation_axis_matrix @ controller_relative_rotation.as_rotvec()
    target_rotation = Rotation.from_rotvec(mapped_rotation_vector) * robot_anchor_rotation
    return target_position.tolist(), target_rotation


class RealmanPoseConverter(Node):
    def __init__(self):
        super().__init__("realman_pose_converter")
        self.robot_name = str(self.declare_parameter("robot_name", "left").value)
        self.position_axis_map_text = str(
            self.declare_parameter("position_axis_map", "z,y,-x").value
        )
        self.rotation_axis_map_text = str(
            self.declare_parameter("rotation_axis_map", "-z,y,x").value
        )
        self.translation_reference_frame = str(
            self.declare_parameter(
                "translation_reference_frame", "tracking"
            ).value
        )
        self.controller_forward_axis = str(
            self.declare_parameter("controller_forward_axis", "z").value
        ).lower()
        self.position_axis_matrix = parse_axis_map(self.position_axis_map_text)
        self.rotation_axis_matrix = parse_axis_map(self.rotation_axis_map_text)
        position_delta_in_reference_frame(
            [0.0, 0.0, 0.0],
            Rotation.identity(),
            self.translation_reference_frame,
            self.controller_forward_axis,
        )
        self.enabled = False
        self.controller_anchor_position: Optional[np.ndarray] = None
        self.controller_anchor_rotation: Optional[Rotation] = None
        self.robot_anchor_position: Optional[np.ndarray] = None
        self.robot_anchor_rotation: Optional[Rotation] = None
        self.current_robot_position: Optional[np.ndarray] = None
        self.current_robot_rotation: Optional[Rotation] = None

        self.create_subscription(String, "vr_pose", self._vr_pose_callback, 10)
        self.create_subscription(JointState, "cur_tcp_pose", self._tcp_pose_callback, 10)
        self.command_publisher = self.create_publisher(JointState, "robot_command_servo_p", 1)
        self.create_service(SetBool, "vr_robot_pose_converter", self._enable_callback)
        self.get_logger().info(
            f"RealMan pose converter ready: robot={self.robot_name}, "
            f"position_axis_map={self.position_axis_map_text}, "
            f"rotation_axis_map={self.rotation_axis_map_text}, "
            f"translation_frame={self.translation_reference_frame}, "
            f"forward_axis={self.controller_forward_axis}"
        )

    def _enable_callback(self, request, response):
        self.enabled = bool(request.data)
        self._reset_anchors()
        response.success = True
        response.message = "Enabled RealMan pose converter" if self.enabled else "Disabled RealMan pose converter"
        return response

    def _reset_anchors(self) -> None:
        self.controller_anchor_position = None
        self.controller_anchor_rotation = None
        self.robot_anchor_position = None
        self.robot_anchor_rotation = None

    def _tcp_pose_callback(self, message: JointState) -> None:
        if len(message.position) < 7:
            return
        self.current_robot_position = np.asarray(message.position[:3], dtype=float)
        qw, qx, qy, qz = [float(value) for value in message.position[3:7]]
        self.current_robot_rotation = Rotation.from_quat([qx, qy, qz, qw])

    def _vr_pose_callback(self, message: String) -> None:
        if not self.enabled:
            return
        fields = message.data.split()
        if len(fields) < 6:
            return
        try:
            values = [float(value) for value in fields[:6]]
        except ValueError:
            return
        if self.current_robot_position is None or self.current_robot_rotation is None:
            return

        controller_position = np.asarray(values[:3], dtype=float) * 1000.0
        controller_rotation = pico_euler_to_rotation(values[3:6])
        if self.controller_anchor_position is None:
            self.controller_anchor_position = controller_position.copy()
            self.controller_anchor_rotation = controller_rotation
            self.robot_anchor_position = self.current_robot_position.copy()
            self.robot_anchor_rotation = self.current_robot_rotation
            self.get_logger().warn(
                f"{self.robot_name} anchors captured; "
                f"position={self.position_axis_map_text}, rotation={self.rotation_axis_map_text}"
            )

        target_position, target_rotation = map_relative_pose(
            self.robot_anchor_position,
            self.robot_anchor_rotation,
            self.controller_anchor_position,
            self.controller_anchor_rotation,
            controller_position,
            controller_rotation,
            self.position_axis_matrix,
            self.rotation_axis_matrix,
            self.translation_reference_frame,
            self.controller_forward_axis,
        )
        target_euler = target_rotation.as_euler("xyz").tolist()
        command = JointState()
        command.header.stamp = self.get_clock().now().to_msg()
        command.position = target_position + target_euler
        self.command_publisher.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = RealmanPoseConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
