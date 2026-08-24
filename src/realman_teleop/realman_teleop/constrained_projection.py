from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class JointConstrainedProjectionResult:
    desired_twist: np.ndarray
    safe_twist: np.ndarray
    joint_velocity: np.ndarray
    lower_velocity_bounds: np.ndarray
    upper_velocity_bounds: np.ndarray
    active_lower_limits: np.ndarray
    active_upper_limits: np.ndarray
    retained_ratio: float
    solver_cost: float
    solver_optimality: float


def joint_velocity_bounds(
    joint_positions: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    velocity_limits: Sequence[float],
    *,
    position_margin_rad: float = 0.05,
    prediction_horizon_s: float = 0.25,
):
    positions = np.asarray(joint_positions, dtype=float)
    lower = np.asarray(lower_limits, dtype=float)
    upper = np.asarray(upper_limits, dtype=float)
    velocity = np.asarray(velocity_limits, dtype=float)
    if not (positions.shape == lower.shape == upper.shape == velocity.shape):
        raise ValueError("joint arrays must have equal shapes")
    if prediction_horizon_s <= 0:
        raise ValueError("prediction_horizon_s must be positive")
    if position_margin_rad < 0:
        raise ValueError("position_margin_rad must be non-negative")
    if np.any(velocity <= 0):
        raise ValueError("velocity limits must be positive")

    safe_lower = lower + position_margin_rad
    safe_upper = upper - position_margin_rad
    if np.any(safe_lower >= safe_upper):
        raise ValueError("joint position margin leaves no feasible range")
    lower_velocity = np.maximum(
        -velocity,
        (safe_lower - positions) / prediction_horizon_s,
    )
    upper_velocity = np.minimum(
        velocity,
        (safe_upper - positions) / prediction_horizon_s,
    )
    if np.any(lower_velocity > upper_velocity):
        raise ValueError("current joint state is outside the configured safe range")
    return lower_velocity, upper_velocity


def project_twist_with_joint_limits(
    jacobian: np.ndarray,
    desired_twist: Sequence[float],
    joint_positions: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    velocity_limits: Sequence[float],
    *,
    position_margin_rad: float = 0.05,
    prediction_horizon_s: float = 0.25,
    damping: float = 0.02,
    characteristic_length_m: float = 0.3,
) -> JointConstrainedProjectionResult:
    matrix = np.asarray(jacobian, dtype=float)
    twist = np.asarray(desired_twist, dtype=float)
    positions = np.asarray(joint_positions, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 6:
        raise ValueError(f"jacobian must have shape (6, n), got {matrix.shape}")
    if twist.shape != (6,):
        raise ValueError(f"desired_twist must have shape (6,), got {twist.shape}")
    if positions.shape != (matrix.shape[1],):
        raise ValueError("joint_positions length must match jacobian columns")
    if damping < 0:
        raise ValueError("damping must be non-negative")
    if characteristic_length_m <= 0:
        raise ValueError("characteristic_length_m must be positive")

    lower_velocity, upper_velocity = joint_velocity_bounds(
        positions,
        lower_limits,
        upper_limits,
        velocity_limits,
        position_margin_rad=position_margin_rad,
        prediction_horizon_s=prediction_horizon_s,
    )
    task_scale = np.diag(
        [1.0, 1.0, 1.0, characteristic_length_m, characteristic_length_m, characteristic_length_m]
    )
    scaled_jacobian = task_scale @ matrix
    scaled_twist = task_scale @ twist
    augmented_matrix = np.vstack(
        [scaled_jacobian, damping * np.eye(matrix.shape[1])]
    )
    augmented_target = np.concatenate(
        [scaled_twist, np.zeros(matrix.shape[1])]
    )
    solution = lsq_linear(
        augmented_matrix,
        augmented_target,
        bounds=(lower_velocity, upper_velocity),
        method="trf",
        lsmr_tol="auto",
        max_iter=100,
    )
    if not solution.success:
        raise RuntimeError(f"bounded least-squares failed: {solution.message}")
    joint_velocity = solution.x
    safe_twist = matrix @ joint_velocity
    desired_norm = float(np.linalg.norm(scaled_twist))
    retained_ratio = (
        1.0
        if desired_norm <= 1e-12
        else float(np.linalg.norm(task_scale @ safe_twist) / desired_norm)
    )
    tolerance = 1e-6
    return JointConstrainedProjectionResult(
        desired_twist=twist,
        safe_twist=safe_twist,
        joint_velocity=joint_velocity,
        lower_velocity_bounds=lower_velocity,
        upper_velocity_bounds=upper_velocity,
        active_lower_limits=joint_velocity <= lower_velocity + tolerance,
        active_upper_limits=joint_velocity >= upper_velocity - tolerance,
        retained_ratio=max(0.0, min(1.0, retained_ratio)),
        solver_cost=float(solution.cost),
        solver_optimality=float(solution.optimality),
    )
