import math
from typing import Iterable, List, Sequence, Tuple

from scipy.spatial.transform import Rotation


def finite_pose(values: Iterable[float]) -> List[float]:
    pose = [float(value) for value in values]
    if len(pose) != 6:
        raise ValueError(f"expected 6 pose values, got {len(pose)}")
    if not all(math.isfinite(value) for value in pose):
        raise ValueError("pose contains a non-finite value")
    return pose


def tamen_mm_pose_to_realman(values: Iterable[float]) -> List[float]:
    pose = finite_pose(values)
    return [pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0, *pose[3:]]


def wrapped_angle_delta(target: float, current: float) -> float:
    return math.atan2(math.sin(target - current), math.cos(target - current))


def _clamp_norm(values: Sequence[float], maximum: float) -> Tuple[List[float], bool]:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= maximum or norm == 0:
        return list(values), False
    scale = maximum / norm
    return [value * scale for value in values], True


def clamp_pose_step(
    current: Sequence[float],
    target: Sequence[float],
    max_position_step_m: float,
    max_rotation_step_rad: float,
) -> Tuple[List[float], bool]:
    current_pose = finite_pose(current)
    target_pose = finite_pose(target)
    position_delta = [target_pose[index] - current_pose[index] for index in range(3)]
    position_delta, position_clamped = _clamp_norm(position_delta, max_position_step_m)
    current_rotation = Rotation.from_euler("xyz", current_pose[3:6])
    target_rotation = Rotation.from_euler("xyz", target_pose[3:6])
    rotation_vector = (target_rotation * current_rotation.inv()).as_rotvec()
    rotation_delta, rotation_clamped = _clamp_norm(rotation_vector, max_rotation_step_rad)
    next_rotation = Rotation.from_rotvec(rotation_delta) * current_rotation
    next_pose = [
        current_pose[index] + position_delta[index] for index in range(3)
    ] + next_rotation.as_euler("xyz").tolist()
    return next_pose, position_clamped or rotation_clamped


def _finite_vector3(values: Sequence[float], name: str) -> List[float]:
    vector = [float(value) for value in values]
    if len(vector) != 3:
        raise ValueError(f"{name} must contain three values")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} contains a non-finite value")
    return vector


def _acceleration_limited_velocity(
    error: Sequence[float],
    velocity: Sequence[float],
    maximum_velocity: float,
    maximum_acceleration: float,
    dt: float,
) -> Tuple[List[float], bool]:
    distance = math.sqrt(sum(value * value for value in error))
    if distance == 0.0:
        desired_velocity = [0.0, 0.0, 0.0]
    else:
        stopping_velocity = math.sqrt(2.0 * maximum_acceleration * distance)
        speed = min(maximum_velocity, stopping_velocity)
        desired_velocity = [value * speed / distance for value in error]

    velocity_change = [
        desired_velocity[index] - velocity[index] for index in range(3)
    ]
    velocity_change, acceleration_limited = _clamp_norm(
        velocity_change,
        maximum_acceleration * dt,
    )
    next_velocity = [
        velocity[index] + velocity_change[index] for index in range(3)
    ]
    return next_velocity, acceleration_limited or _stopping_velocity_limited(
        distance, maximum_velocity, maximum_acceleration
    )


def _stopping_velocity_limited(
    distance: float,
    maximum_velocity: float,
    maximum_acceleration: float,
) -> bool:
    return distance > 0.0 and math.sqrt(
        2.0 * maximum_acceleration * distance
    ) < maximum_velocity


def acceleration_limit_pose(
    current: Sequence[float],
    target: Sequence[float],
    linear_velocity: Sequence[float],
    angular_velocity: Sequence[float],
    dt: float,
    max_linear_velocity_mps: float,
    max_linear_acceleration_mps2: float,
    max_angular_velocity_radps: float,
    max_angular_acceleration_radps2: float,
) -> Tuple[List[float], List[float], List[float], bool]:
    """Advance a Cartesian pose while keeping velocity changes continuous."""
    current_pose = finite_pose(current)
    target_pose = finite_pose(target)
    current_linear_velocity = _finite_vector3(linear_velocity, "linear_velocity")
    current_angular_velocity = _finite_vector3(angular_velocity, "angular_velocity")
    limits = (
        dt,
        max_linear_velocity_mps,
        max_linear_acceleration_mps2,
        max_angular_velocity_radps,
        max_angular_acceleration_radps2,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in limits):
        raise ValueError("pose rate limits and dt must be finite and positive")

    position_error = [target_pose[index] - current_pose[index] for index in range(3)]
    next_linear_velocity, position_limited = _acceleration_limited_velocity(
        position_error,
        current_linear_velocity,
        max_linear_velocity_mps,
        max_linear_acceleration_mps2,
        dt,
    )
    position_step = [value * dt for value in next_linear_velocity]
    position_distance = math.sqrt(sum(value * value for value in position_error))
    position_step_distance = math.sqrt(sum(value * value for value in position_step))
    position_speed = math.sqrt(sum(value * value for value in next_linear_velocity))
    position_toward_target = sum(
        position_step[index] * position_error[index] for index in range(3)
    ) > 0.0
    if (
        position_step_distance >= position_distance
        and position_distance > 0.0
        and position_toward_target
        and position_speed <= max_linear_acceleration_mps2 * dt
    ):
        position_step = position_error
        next_linear_velocity = [value / dt for value in position_step]

    current_rotation = Rotation.from_euler("xyz", current_pose[3:6])
    target_rotation = Rotation.from_euler("xyz", target_pose[3:6])
    rotation_error = (target_rotation * current_rotation.inv()).as_rotvec().tolist()
    next_angular_velocity, rotation_limited = _acceleration_limited_velocity(
        rotation_error,
        current_angular_velocity,
        max_angular_velocity_radps,
        max_angular_acceleration_radps2,
        dt,
    )
    rotation_step = [value * dt for value in next_angular_velocity]
    rotation_distance = math.sqrt(sum(value * value for value in rotation_error))
    rotation_step_distance = math.sqrt(sum(value * value for value in rotation_step))
    angular_speed = math.sqrt(sum(value * value for value in next_angular_velocity))
    rotation_toward_target = sum(
        rotation_step[index] * rotation_error[index] for index in range(3)
    ) > 0.0
    if (
        rotation_step_distance >= rotation_distance
        and rotation_distance > 0.0
        and rotation_toward_target
        and angular_speed <= max_angular_acceleration_radps2 * dt
    ):
        rotation_step = rotation_error
        next_angular_velocity = [value / dt for value in rotation_step]
    next_rotation = Rotation.from_rotvec(rotation_step) * current_rotation

    next_pose = [
        current_pose[index] + position_step[index] for index in range(3)
    ] + next_rotation.as_euler("xyz").tolist()
    return (
        next_pose,
        next_linear_velocity,
        next_angular_velocity,
        position_limited or rotation_limited,
    )


def select_pose_target(
    anchor: Sequence[float],
    target: Sequence[float],
    position_scale: float,
    rotation_scale: float,
    direct_passthrough: bool,
) -> List[float]:
    if direct_passthrough:
        return finite_pose(target)
    return scale_pose_from_anchor(anchor, target, position_scale, rotation_scale)


def select_next_pose(
    current: Sequence[float],
    target: Sequence[float],
    max_position_step_m: float,
    max_rotation_step_rad: float,
    direct_passthrough: bool,
) -> Tuple[List[float], bool]:
    if direct_passthrough:
        return finite_pose(target), False
    return clamp_pose_step(
        current,
        target,
        max_position_step_m,
        max_rotation_step_rad,
    )


def pose_delta_norms(reference: Sequence[float], target: Sequence[float]) -> Tuple[float, float]:
    reference_pose = finite_pose(reference)
    target_pose = finite_pose(target)
    position_delta = [target_pose[index] - reference_pose[index] for index in range(3)]
    reference_rotation = Rotation.from_euler("xyz", reference_pose[3:6])
    target_rotation = Rotation.from_euler("xyz", target_pose[3:6])
    return (
        math.sqrt(sum(value * value for value in position_delta)),
        (target_rotation * reference_rotation.inv()).magnitude(),
    )


def scale_pose_from_anchor(
    anchor: Sequence[float],
    target: Sequence[float],
    position_scale: float,
    rotation_scale: float,
) -> List[float]:
    anchor_pose = finite_pose(anchor)
    target_pose = finite_pose(target)
    if not math.isfinite(position_scale) or position_scale < 0:
        raise ValueError("position_scale must be finite and non-negative")
    if not math.isfinite(rotation_scale) or rotation_scale < 0:
        raise ValueError("rotation_scale must be finite and non-negative")

    scaled_position = [
        anchor_pose[index]
        + (target_pose[index] - anchor_pose[index]) * position_scale
        for index in range(3)
    ]
    anchor_rotation = Rotation.from_euler("xyz", anchor_pose[3:6])
    target_rotation = Rotation.from_euler("xyz", target_pose[3:6])
    relative_vector = (target_rotation * anchor_rotation.inv()).as_rotvec()
    scaled_rotation = Rotation.from_rotvec(relative_vector * rotation_scale) * anchor_rotation
    return scaled_position + scaled_rotation.as_euler("xyz").tolist()


def realman_quaternion_pose(values: Iterable[float]) -> List[float]:
    pose = finite_pose(values)
    quaternion = Rotation.from_euler("xyz", pose[3:6]).as_quat()
    return [pose[0], pose[1], pose[2], quaternion[3], quaternion[0], quaternion[1], quaternion[2]]


def realman_canfd_pose(values: Iterable[float], pose_format: str) -> List[float]:
    pose = finite_pose(values)
    if pose_format == "euler":
        return pose
    if pose_format == "quaternion":
        return realman_quaternion_pose(pose)
    raise ValueError("CANFD pose format must be 'euler' or 'quaternion'")


def inside_workspace(anchor: Sequence[float], target: Sequence[float], limit_m: float) -> bool:
    if limit_m <= 0:
        raise ValueError("workspace limit must be positive")
    anchor_pose = finite_pose(anchor)
    target_pose = finite_pose(target)
    return all(abs(target_pose[index] - anchor_pose[index]) <= limit_m for index in range(3))


def project_pose_to_workspace(
    anchor: Sequence[float],
    target: Sequence[float],
    position_limit_m: float,
    rotation_limit_rad: float,
) -> Tuple[List[float], bool, bool]:
    if position_limit_m <= 0:
        raise ValueError("position_limit_m must be positive")
    if rotation_limit_rad <= 0:
        raise ValueError("rotation_limit_rad must be positive")
    anchor_pose = finite_pose(anchor)
    target_pose = finite_pose(target)

    projected_position = []
    position_limited = False
    for index in range(3):
        delta = target_pose[index] - anchor_pose[index]
        limited_delta = max(-position_limit_m, min(position_limit_m, delta))
        position_limited = position_limited or not math.isclose(delta, limited_delta, abs_tol=1e-12)
        projected_position.append(anchor_pose[index] + limited_delta)

    anchor_rotation = Rotation.from_euler("xyz", anchor_pose[3:6])
    target_rotation = Rotation.from_euler("xyz", target_pose[3:6])
    relative_vector = (target_rotation * anchor_rotation.inv()).as_rotvec()
    rotation_norm = math.sqrt(sum(float(value) ** 2 for value in relative_vector))
    rotation_limited = rotation_norm > rotation_limit_rad
    if rotation_limited and rotation_norm > 0:
        relative_vector = relative_vector * (rotation_limit_rad / rotation_norm)
    projected_rotation = Rotation.from_rotvec(relative_vector) * anchor_rotation
    return (
        projected_position + projected_rotation.as_euler("xyz").tolist(),
        position_limited,
        rotation_limited,
    )


def arm_error_active(error: object) -> bool:
    if not isinstance(error, dict) or not error:
        return False
    values = error.get("err", error.get("code", error.get("status", [])))
    if not isinstance(values, (list, tuple)):
        values = [values]
    for value in values:
        if value in (None, False, ""):
            continue
        try:
            if int(str(value), 0) != 0:
                return True
        except ValueError:
            return True
    return False
