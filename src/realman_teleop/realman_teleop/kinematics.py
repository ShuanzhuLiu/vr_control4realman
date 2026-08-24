import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def _vector(text: str, length: int) -> np.ndarray:
    values = [float(value) for value in text.split()]
    if len(values) != length:
        raise ValueError(f"expected {length} values, got {values}")
    return np.asarray(values, dtype=float)


def _transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(np.asarray(axis, dtype=float) * angle).as_matrix()
    return transform


def pose_to_transform(pose: Sequence[float]) -> np.ndarray:
    values = np.asarray(pose, dtype=float)
    if values.shape != (6,):
        raise ValueError(f"expected 6 pose values, got {values.shape}")
    return _transform(values[:3], values[3:6])


def quaternion_pose_to_transform(pose: Sequence[float]) -> np.ndarray:
    values = np.asarray(pose, dtype=float)
    if values.shape != (7,):
        raise ValueError(f"expected 7 pose values, got {values.shape}")
    transform = np.eye(4)
    transform[:3, 3] = values[:3]
    transform[:3, :3] = Rotation.from_quat(
        [values[4], values[5], values[6], values[3]]
    ).as_matrix()
    return transform


@dataclass(frozen=True)
class ChainJoint:
    name: str
    parent: str
    child: str
    joint_type: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    velocity: float


@dataclass(frozen=True)
class KinematicMetrics:
    singular_values: np.ndarray
    sigma_min: float
    condition_number: float
    manipulability: float
    joint_limit_margins: np.ndarray
    normalized_joint_limit_margins: np.ndarray

    @property
    def minimum_joint_limit_margin(self) -> float:
        return float(np.min(self.joint_limit_margins))

    @property
    def minimum_normalized_joint_limit_margin(self) -> float:
        return float(np.min(self.normalized_joint_limit_margins))


class SerialChainKinematics:
    def __init__(self, joints: List[ChainJoint], base_link: str, tip_link: str):
        self.joints = joints
        self.base_link = base_link
        self.tip_link = tip_link
        self.actuated_joints = [joint for joint in joints if joint.joint_type != "fixed"]
        self.joint_names = [joint.name for joint in self.actuated_joints]
        self.lower_limits = np.asarray([joint.lower for joint in self.actuated_joints])
        self.upper_limits = np.asarray([joint.upper for joint in self.actuated_joints])
        self.velocity_limits = np.asarray([joint.velocity for joint in self.actuated_joints])

    @classmethod
    def from_urdf(cls, path: Path, base_link: str, tip_link: str):
        root = ElementTree.parse(path).getroot()
        by_child = {}
        for joint_element in root.findall("joint"):
            child = joint_element.find("child").attrib["link"]
            parent = joint_element.find("parent").attrib["link"]
            origin_element = joint_element.find("origin")
            axis_element = joint_element.find("axis")
            limit_element = joint_element.find("limit")
            origin_xyz = _vector(
                origin_element.attrib.get("xyz", "0 0 0") if origin_element is not None else "0 0 0",
                3,
            )
            origin_rpy = _vector(
                origin_element.attrib.get("rpy", "0 0 0") if origin_element is not None else "0 0 0",
                3,
            )
            axis = _vector(
                axis_element.attrib.get("xyz", "0 0 1") if axis_element is not None else "0 0 1",
                3,
            )
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm == 0:
                raise ValueError(f"joint {joint_element.attrib['name']} has a zero axis")
            axis /= axis_norm
            joint_type = joint_element.attrib.get("type", "fixed")
            lower = float(limit_element.attrib.get("lower", "-inf")) if limit_element is not None else -math.inf
            upper = float(limit_element.attrib.get("upper", "inf")) if limit_element is not None else math.inf
            velocity = float(limit_element.attrib.get("velocity", "inf")) if limit_element is not None else math.inf
            by_child[child] = ChainJoint(
                name=joint_element.attrib["name"],
                parent=parent,
                child=child,
                joint_type=joint_type,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
                lower=lower,
                upper=upper,
                velocity=velocity,
            )

        chain = []
        current = tip_link
        while current != base_link:
            if current not in by_child:
                raise ValueError(f"no joint connects {current} to base link {base_link}")
            joint = by_child[current]
            chain.append(joint)
            current = joint.parent
        chain.reverse()
        return cls(chain, base_link, tip_link)

    def _validate_positions(self, positions: Sequence[float]) -> np.ndarray:
        values = np.asarray(positions, dtype=float)
        if values.shape != (len(self.actuated_joints),):
            raise ValueError(
                f"expected {len(self.actuated_joints)} joint positions, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("joint positions contain non-finite values")
        return values

    def forward(self, positions: Sequence[float]) -> np.ndarray:
        values = self._validate_positions(positions)
        transform = np.eye(4)
        index = 0
        for joint in self.joints:
            transform = transform @ _transform(joint.origin_xyz, joint.origin_rpy)
            if joint.joint_type in ("revolute", "continuous"):
                transform = transform @ _axis_rotation(joint.axis, values[index])
                index += 1
            elif joint.joint_type == "prismatic":
                translation = np.eye(4)
                translation[:3, 3] = joint.axis * values[index]
                transform = transform @ translation
                index += 1
        return transform

    def jacobian(self, positions: Sequence[float]) -> np.ndarray:
        values = self._validate_positions(positions)
        transform = np.eye(4)
        axes_world = []
        origins_world = []
        index = 0
        for joint in self.joints:
            transform = transform @ _transform(joint.origin_xyz, joint.origin_rpy)
            if joint.joint_type in ("revolute", "continuous", "prismatic"):
                origins_world.append(transform[:3, 3].copy())
                axes_world.append(transform[:3, :3] @ joint.axis)
                if joint.joint_type in ("revolute", "continuous"):
                    transform = transform @ _axis_rotation(joint.axis, values[index])
                else:
                    translation = np.eye(4)
                    translation[:3, 3] = joint.axis * values[index]
                    transform = transform @ translation
                index += 1

        end_position = transform[:3, 3]
        jacobian = np.zeros((6, len(self.actuated_joints)))
        for column, joint in enumerate(self.actuated_joints):
            axis = axes_world[column]
            if joint.joint_type in ("revolute", "continuous"):
                jacobian[:3, column] = np.cross(axis, end_position - origins_world[column])
                jacobian[3:, column] = axis
            else:
                jacobian[:3, column] = axis
        return jacobian

    def frame_origins(self, positions: Sequence[float]) -> List[np.ndarray]:
        values = self._validate_positions(positions)
        transform = self.base_transform.copy()
        points = [transform[:3, 3].copy()]
        for index in range(self.dof):
            alpha_rotation = np.eye(4)
            alpha_rotation[:3, :3] = Rotation.from_euler(
                "x", self.alpha[index]
            ).as_matrix()
            x_translation = np.eye(4)
            x_translation[0, 3] = self.a[index]
            joint_frame = transform @ alpha_rotation @ x_translation
            points.append(joint_frame[:3, 3].copy())
            transform = transform @ self._row_transform(
                self.a[index],
                self.alpha[index],
                self.d[index],
                values[index] + self.offset[index],
            )
        points.append((transform @ self.tool_transform)[:3, 3].copy())
        return points

    def metrics(self, positions: Sequence[float]) -> KinematicMetrics:
        values = self._validate_positions(positions)
        jacobian = self.jacobian(values)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        sigma_min = float(np.min(singular_values))
        sigma_max = float(np.max(singular_values))
        condition_number = math.inf if sigma_min <= 1e-12 else sigma_max / sigma_min
        manipulability = float(np.prod(singular_values))
        lower_margins = values - self.lower_limits
        upper_margins = self.upper_limits - values
        joint_limit_margins = np.minimum(lower_margins, upper_margins)
        ranges = self.upper_limits - self.lower_limits
        normalized = joint_limit_margins / np.maximum(ranges, 1e-12)
        return KinematicMetrics(
            singular_values=singular_values,
            sigma_min=sigma_min,
            condition_number=condition_number,
            manipulability=manipulability,
            joint_limit_margins=joint_limit_margins,
            normalized_joint_limit_margins=normalized,
        )


class ModifiedDhKinematics:
    def __init__(
        self,
        a: Sequence[float],
        alpha_rad: Sequence[float],
        d: Sequence[float],
        offset_rad: Sequence[float],
        lower_limits_rad: Sequence[float],
        upper_limits_rad: Sequence[float],
        velocity_limits_rad_s: Sequence[float],
        tool_transform: np.ndarray = None,
        base_transform: np.ndarray = None,
    ):
        self.a = np.asarray(a, dtype=float)
        self.alpha = np.asarray(alpha_rad, dtype=float)
        self.d = np.asarray(d, dtype=float)
        self.offset = np.asarray(offset_rad, dtype=float)
        self.lower_limits = np.asarray(lower_limits_rad, dtype=float)
        self.upper_limits = np.asarray(upper_limits_rad, dtype=float)
        self.velocity_limits = np.asarray(velocity_limits_rad_s, dtype=float)
        lengths = {
            len(self.a),
            len(self.alpha),
            len(self.d),
            len(self.offset),
            len(self.lower_limits),
            len(self.upper_limits),
            len(self.velocity_limits),
        }
        if len(lengths) != 1:
            raise ValueError("Modified DH arrays must have equal lengths")
        self.dof = len(self.a)
        self.tool_transform = (
            np.eye(4) if tool_transform is None else np.asarray(tool_transform, dtype=float)
        )
        self.base_transform = (
            np.eye(4) if base_transform is None else np.asarray(base_transform, dtype=float)
        )
        if self.tool_transform.shape != (4, 4) or self.base_transform.shape != (4, 4):
            raise ValueError("base and tool transforms must be 4x4")

    @classmethod
    def from_client(cls, client):
        dh = client.raw_call("rm_algo_get_dh", check=False)
        tool = client.raw_call("rm_algo_get_curr_toolframe", check=False)
        lower_deg, upper_deg = client.get_joint_position_limits()
        arm = client._require_arm()
        velocity_deg = arm.rm_algo_get_joint_max_speed()
        model = cls(
            a=dh["a"],
            alpha_rad=np.radians(dh["alpha"]),
            d=dh["d"],
            offset_rad=np.radians(dh["offset"]),
            lower_limits_rad=np.radians(lower_deg),
            upper_limits_rad=np.radians(upper_deg),
            velocity_limits_rad_s=np.radians(velocity_deg),
            tool_transform=pose_to_transform(tool["pose"]),
        )
        zero_sdk_pose = client.raw_call(
            "rm_algo_forward_kinematics", [0.0] * model.dof, 0
        )
        zero_sdk_transform = quaternion_pose_to_transform(zero_sdk_pose)
        model.base_transform = zero_sdk_transform @ np.linalg.inv(
            model.forward([0.0] * model.dof)
        )
        return model

    def _validate_positions(self, positions: Sequence[float]) -> np.ndarray:
        values = np.asarray(positions, dtype=float)
        if values.shape != (self.dof,):
            raise ValueError(f"expected {self.dof} joint positions, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("joint positions contain non-finite values")
        return values

    @staticmethod
    def _row_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        cosine_theta = math.cos(theta)
        sine_theta = math.sin(theta)
        cosine_alpha = math.cos(alpha)
        sine_alpha = math.sin(alpha)
        return np.asarray(
            [
                [cosine_theta, -sine_theta, 0.0, a],
                [sine_theta * cosine_alpha, cosine_theta * cosine_alpha, -sine_alpha, -d * sine_alpha],
                [sine_theta * sine_alpha, cosine_theta * sine_alpha, cosine_alpha, d * cosine_alpha],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def forward(self, positions: Sequence[float]) -> np.ndarray:
        values = self._validate_positions(positions)
        transform = self.base_transform.copy()
        for index in range(self.dof):
            transform = transform @ self._row_transform(
                self.a[index],
                self.alpha[index],
                self.d[index],
                values[index] + self.offset[index],
            )
        return transform @ self.tool_transform

    def jacobian(self, positions: Sequence[float]) -> np.ndarray:
        values = self._validate_positions(positions)
        transform = self.base_transform.copy()
        axes_world = []
        origins_world = []
        for index in range(self.dof):
            alpha_rotation = np.eye(4)
            alpha_rotation[:3, :3] = Rotation.from_euler(
                "x", self.alpha[index]
            ).as_matrix()
            x_translation = np.eye(4)
            x_translation[0, 3] = self.a[index]
            joint_frame = transform @ alpha_rotation @ x_translation
            axes_world.append(joint_frame[:3, :3] @ np.array([0.0, 0.0, 1.0]))
            origins_world.append(joint_frame[:3, 3].copy())
            transform = transform @ self._row_transform(
                self.a[index],
                self.alpha[index],
                self.d[index],
                values[index] + self.offset[index],
            )

        end_transform = transform @ self.tool_transform
        end_position = end_transform[:3, 3]
        jacobian = np.zeros((6, self.dof))
        for column, axis in enumerate(axes_world):
            jacobian[:3, column] = np.cross(axis, end_position - origins_world[column])
            jacobian[3:, column] = axis
        return jacobian

    def metrics(self, positions: Sequence[float]) -> KinematicMetrics:
        values = self._validate_positions(positions)
        singular_values = np.linalg.svd(self.jacobian(values), compute_uv=False)
        sigma_min = float(np.min(singular_values))
        sigma_max = float(np.max(singular_values))
        condition_number = math.inf if sigma_min <= 1e-12 else sigma_max / sigma_min
        margins = np.minimum(values - self.lower_limits, self.upper_limits - values)
        ranges = self.upper_limits - self.lower_limits
        return KinematicMetrics(
            singular_values=singular_values,
            sigma_min=sigma_min,
            condition_number=condition_number,
            manipulability=float(np.prod(singular_values)),
            joint_limit_margins=margins,
            normalized_joint_limit_margins=margins / np.maximum(ranges, 1e-12),
        )
