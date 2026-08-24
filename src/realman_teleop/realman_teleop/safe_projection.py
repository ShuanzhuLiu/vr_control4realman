from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SafeProjectionResult:
    desired_twist: np.ndarray
    safe_twist: np.ndarray
    blocked_twist: np.ndarray
    singular_values: np.ndarray
    direction_gains: np.ndarray
    projector: np.ndarray
    retained_ratio: float

    @property
    def blocked_ratio(self) -> float:
        return 1.0 - self.retained_ratio


def smooth_singularity_gain(
    singular_value: float,
    hard_threshold: float,
    soft_threshold: float,
) -> float:
    if hard_threshold < 0 or soft_threshold <= hard_threshold:
        raise ValueError("thresholds must satisfy 0 <= hard < soft")
    value = float(singular_value)
    if value <= hard_threshold:
        return 0.0
    if value >= soft_threshold:
        return 1.0
    normalized = (value - hard_threshold) / (soft_threshold - hard_threshold)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def project_twist_to_safe_subspace(
    jacobian: np.ndarray,
    desired_twist: Sequence[float],
    *,
    hard_threshold: float = 0.02,
    soft_threshold: float = 0.08,
    characteristic_length_m: float = 0.3,
) -> SafeProjectionResult:
    matrix = np.asarray(jacobian, dtype=float)
    twist = np.asarray(desired_twist, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 6:
        raise ValueError(f"jacobian must have shape (6, n), got {matrix.shape}")
    if twist.shape != (6,):
        raise ValueError(f"desired_twist must have shape (6,), got {twist.shape}")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(twist)):
        raise ValueError("jacobian and twist must be finite")
    if characteristic_length_m <= 0:
        raise ValueError("characteristic_length_m must be positive")

    task_scale = np.diag(
        [1.0, 1.0, 1.0, characteristic_length_m, characteristic_length_m, characteristic_length_m]
    )
    inverse_scale = np.diag(
        [1.0, 1.0, 1.0, 1.0 / characteristic_length_m, 1.0 / characteristic_length_m, 1.0 / characteristic_length_m]
    )
    scaled_jacobian = task_scale @ matrix
    scaled_twist = task_scale @ twist
    left_vectors, singular_values, _ = np.linalg.svd(scaled_jacobian, full_matrices=True)
    gains = np.zeros(6)
    gains[: len(singular_values)] = [
        smooth_singularity_gain(value, hard_threshold, soft_threshold)
        for value in singular_values
    ]
    scaled_projector = left_vectors @ np.diag(gains) @ left_vectors.T
    safe_twist = inverse_scale @ scaled_projector @ scaled_twist
    blocked_twist = twist - safe_twist
    desired_norm = float(np.linalg.norm(scaled_twist))
    retained_ratio = (
        1.0
        if desired_norm <= 1e-12
        else float(np.linalg.norm(task_scale @ safe_twist) / desired_norm)
    )
    projector = inverse_scale @ scaled_projector @ task_scale
    return SafeProjectionResult(
        desired_twist=twist,
        safe_twist=safe_twist,
        blocked_twist=blocked_twist,
        singular_values=singular_values,
        direction_gains=gains,
        projector=projector,
        retained_ratio=max(0.0, min(1.0, retained_ratio)),
    )


def pose_delta_twist(reference_pose: Sequence[float], target_pose: Sequence[float]) -> np.ndarray:
    reference = np.asarray(reference_pose, dtype=float)
    target = np.asarray(target_pose, dtype=float)
    if reference.shape != (6,) or target.shape != (6,):
        raise ValueError("reference_pose and target_pose must have shape (6,)")
    reference_rotation = Rotation.from_euler("xyz", reference[3:6])
    target_rotation = Rotation.from_euler("xyz", target[3:6])
    return np.concatenate(
        [
            target[:3] - reference[:3],
            (target_rotation * reference_rotation.inv()).as_rotvec(),
        ]
    )


def apply_pose_delta_twist(reference_pose: Sequence[float], delta_twist: Sequence[float]):
    reference = np.asarray(reference_pose, dtype=float)
    delta = np.asarray(delta_twist, dtype=float)
    if reference.shape != (6,) or delta.shape != (6,):
        raise ValueError("reference_pose and delta_twist must have shape (6,)")
    reference_rotation = Rotation.from_euler("xyz", reference[3:6])
    target_rotation = Rotation.from_rotvec(delta[3:6]) * reference_rotation
    return np.concatenate(
        [reference[:3] + delta[:3], target_rotation.as_euler("xyz")]
    ).tolist()
