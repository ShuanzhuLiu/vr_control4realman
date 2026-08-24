from dataclasses import dataclass
from typing import List

import numpy as np


SINGULARITY_LABELS = {
    0: "normal",
    -1: "shoulder",
    -2: "elbow",
    -3: "wrist",
}


@dataclass(frozen=True)
class NativeSafetyMetrics:
    analytic_singularity_code: int
    analytic_singularity_label: str
    shoulder_plane_distance_m: float
    universal_singularity_code: int
    universal_singular: bool
    self_collision_code: int
    self_collision: bool
    joint_limit_margins_deg: List[float]
    minimum_joint_limit_margin_deg: float

    @property
    def unsafe(self) -> bool:
        return (
            self.analytic_singularity_code != 0
            or self.universal_singular
            or self.self_collision
            or self.minimum_joint_limit_margin_deg < 0
        )


class RealmanNativeSafetyAnalyzer:
    def __init__(self, client, singular_value_threshold: float = 0.01):
        if singular_value_threshold <= 0 or singular_value_threshold >= 1:
            raise ValueError("singular_value_threshold must be in (0, 1)")
        self.client = client
        self.singular_value_threshold = float(singular_value_threshold)
        lower, upper = client.get_joint_position_limits()
        self.lower_limits_deg = np.asarray(lower, dtype=float)
        self.upper_limits_deg = np.asarray(upper, dtype=float)

    def analyze(self, joints_deg) -> NativeSafetyMetrics:
        joints = np.asarray(joints_deg, dtype=float)
        if joints.shape != self.lower_limits_deg.shape:
            raise ValueError(
                f"expected {len(self.lower_limits_deg)} joints, got {joints.shape}"
            )
        if not np.all(np.isfinite(joints)):
            raise ValueError("joint values contain non-finite values")

        analytic_result = self.client.raw_call(
            "rm_algo_kin_robot_singularity_analyse", joints.tolist(), check=False
        )
        if not isinstance(analytic_result, tuple) or len(analytic_result) != 2:
            raise RuntimeError(f"unexpected analytic singularity result: {analytic_result!r}")
        analytic_code = int(analytic_result[0])
        shoulder_distance = float(analytic_result[1])

        universal_code = int(
            self.client.raw_call(
                "rm_algo_universal_singularity_analyse",
                joints.tolist(),
                self.singular_value_threshold,
                check=False,
            )
        )
        collision_code = int(
            self.client.raw_call(
                "rm_algo_safety_robot_self_collision_detection",
                joints.tolist(),
                check=False,
            )
        )

        margins = np.minimum(
            joints - self.lower_limits_deg,
            self.upper_limits_deg - joints,
        )
        return NativeSafetyMetrics(
            analytic_singularity_code=analytic_code,
            analytic_singularity_label=SINGULARITY_LABELS.get(
                analytic_code, f"unknown({analytic_code})"
            ),
            shoulder_plane_distance_m=shoulder_distance,
            universal_singularity_code=universal_code,
            universal_singular=universal_code == -1,
            self_collision_code=collision_code,
            self_collision=collision_code != 0,
            joint_limit_margins_deg=margins.tolist(),
            minimum_joint_limit_margin_deg=float(np.min(margins)),
        )
