import math

import pytest

from realman_teleop.safety import (
    acceleration_limit_pose,
    arm_error_active,
    clamp_pose_step,
    inside_workspace,
    pose_delta_norms,
    project_pose_to_workspace,
    realman_canfd_pose,
    realman_quaternion_pose,
    scale_pose_from_anchor,
    select_next_pose,
    select_pose_target,
    tamen_mm_pose_to_realman,
    wrapped_angle_delta,
)


def test_acceleration_limit_pose_ramps_velocity_and_reaches_target():
    current = [0.0] * 6
    target = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
    linear_velocity = [0.0, 0.0, 0.0]
    angular_velocity = [0.0, 0.0, 0.0]

    first, linear_velocity, angular_velocity, limited = acceleration_limit_pose(
        current,
        target,
        linear_velocity,
        angular_velocity,
        dt=0.008,
        max_linear_velocity_mps=0.15,
        max_linear_acceleration_mps2=1.2,
        max_angular_velocity_radps=0.5,
        max_angular_acceleration_radps2=4.0,
    )
    assert limited
    assert first[0] == pytest.approx(0.0000768)
    assert linear_velocity[0] == pytest.approx(0.0096)

    pose = first
    previous_linear_velocity = list(linear_velocity)
    for _ in range(200):
        pose, linear_velocity, angular_velocity, _ = acceleration_limit_pose(
            pose,
            target,
            linear_velocity,
            angular_velocity,
            dt=0.008,
            max_linear_velocity_mps=0.15,
            max_linear_acceleration_mps2=1.2,
            max_angular_velocity_radps=0.5,
            max_angular_acceleration_radps2=4.0,
        )
        velocity_change = math.sqrt(
            sum(
                (linear_velocity[index] - previous_linear_velocity[index]) ** 2
                for index in range(3)
            )
        )
        assert velocity_change <= 1.2 * 0.008 + 1e-12
        previous_linear_velocity = list(linear_velocity)
    assert pose[0] == pytest.approx(target[0], abs=1e-9)
    assert abs(linear_velocity[0]) < 1e-9


def test_acceleration_limit_pose_limits_rotation_acceleration():
    pose, _, angular_velocity, limited = acceleration_limit_pose(
        [0.0] * 6,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        dt=0.01,
        max_linear_velocity_mps=0.15,
        max_linear_acceleration_mps2=1.2,
        max_angular_velocity_radps=0.5,
        max_angular_acceleration_radps2=4.0,
    )
    assert limited
    assert angular_velocity[2] == pytest.approx(0.04)
    assert pose[5] == pytest.approx(0.0004)


def test_tamen_pose_converts_millimeters_to_meters():
    assert tamen_mm_pose_to_realman([100, -200, 300, 0.1, 0.2, 0.3]) == [
        0.1,
        -0.2,
        0.3,
        0.1,
        0.2,
        0.3,
    ]


def test_invalid_pose_is_rejected():
    with pytest.raises(ValueError):
        tamen_mm_pose_to_realman([1, 2, 3])
    with pytest.raises(ValueError):
        tamen_mm_pose_to_realman([1, 2, 3, 4, 5, math.nan])


def test_angle_delta_uses_shortest_path():
    assert wrapped_angle_delta(-math.pi + 0.1, math.pi - 0.1) == pytest.approx(0.2)


def test_pose_step_is_norm_limited():
    next_pose, clamped = clamp_pose_step(
        [0, 0, 0, 0, 0, 0],
        [0.03, 0.04, 0, 0.3, 0.4, 0],
        max_position_step_m=0.01,
        max_rotation_step_rad=0.1,
    )
    assert clamped is True
    assert math.sqrt(sum(value * value for value in next_pose[:3])) == pytest.approx(0.01)
    assert pose_delta_norms([0, 0, 0, 0, 0, 0], next_pose)[1] == pytest.approx(0.1)


def test_workspace_is_relative_to_enable_anchor():
    anchor = [0.2, 0.0, 0.3, 0, 0, 0]
    assert inside_workspace(anchor, [0.25, -0.05, 0.35, 1, 1, 1], 0.1)
    assert not inside_workspace(anchor, [0.31, 0.0, 0.3, 0, 0, 0], 0.1)


def test_initial_delta_norms_include_wrapped_rotation():
    position, rotation = pose_delta_norms(
        [0, 0, 0, 0, 0, math.pi - 0.1],
        [0.01, 0, 0, 0, 0, -math.pi + 0.1],
    )
    assert position == pytest.approx(0.01)
    assert rotation == pytest.approx(0.2)


def test_rotation_near_gimbal_lock_uses_physical_angle():
    anchor = [0.461389, 0.516765, 0.158435, -2.28, 1.549, -0.73]
    target = [0.45034, 0.50172, 0.16712, -3.2283, 1.43908, -1.76525]
    _, physical_rotation = pose_delta_norms(anchor, target)
    assert math.degrees(physical_rotation) == pytest.approx(8.5426, abs=0.001)


def test_realman_quaternion_pose_uses_qw_qx_qy_qz_order():
    pose = realman_quaternion_pose([0.1, 0.2, 0.3, 0.0, 0.0, math.pi])
    assert pose[:3] == [0.1, 0.2, 0.3]
    assert pose[3] == pytest.approx(0.0, abs=1e-7)
    assert pose[4] == pytest.approx(0.0, abs=1e-7)
    assert pose[5] == pytest.approx(0.0, abs=1e-7)
    assert abs(pose[6]) == pytest.approx(1.0)


def test_canfd_euler_pose_preserves_original_six_values():
    pose = [0.1, 0.2, 0.3, -0.4, 0.5, -0.6]
    assert realman_canfd_pose(pose, "euler") == pose


def test_canfd_pose_rejects_unknown_format():
    with pytest.raises(ValueError, match="pose format"):
        realman_canfd_pose([0.0] * 6, "matrix")


def test_pose_mapping_scales_translation_and_physical_rotation():
    anchor = [0.1, 0.2, 0.3, -2.28, 1.549, -0.73]
    target = [0.3, 0.1, 0.5, -3.2283, 1.43908, -1.76525]
    scaled = scale_pose_from_anchor(anchor, target, 0.25, 0.30)
    assert scaled[:3] == pytest.approx([0.15, 0.175, 0.35])
    original_rotation = pose_delta_norms(anchor, target)[1]
    scaled_rotation = pose_delta_norms(anchor, scaled)[1]
    assert scaled_rotation == pytest.approx(original_rotation * 0.30)


def test_pose_mapping_rejects_negative_scales():
    with pytest.raises(ValueError):
        scale_pose_from_anchor([0] * 6, [0] * 6, -0.1, 1.0)


def test_direct_passthrough_preserves_full_mapped_target():
    anchor = [0.4, 0.1, 0.3, 0.2, -0.1, 0.3]
    target = [0.46, 0.04, 0.35, -0.4, 0.5, -0.2]

    selected = select_pose_target(anchor, target, 0.2, 0.05, True)
    next_pose, limited = select_next_pose(
        anchor,
        selected,
        max_position_step_m=0.0006,
        max_rotation_step_rad=0.002,
        direct_passthrough=True,
    )

    assert next_pose == target
    assert not limited


def test_workspace_projection_clips_only_unsafe_position_axes():
    projected, position_limited, rotation_limited = project_pose_to_workspace(
        [0, 0, 0, 0, 0, 0],
        [0.2, 0.03, -0.15, 0, 0, 0],
        position_limit_m=0.08,
        rotation_limit_rad=0.35,
    )
    assert projected[:3] == pytest.approx([0.08, 0.03, -0.08])
    assert position_limited is True
    assert rotation_limited is False


def test_workspace_projection_limits_rotation_without_changing_position():
    projected, position_limited, rotation_limited = project_pose_to_workspace(
        [0, 0, 0, 0, 0, 0],
        [0.01, 0.02, 0.03, 0, 0, 1.0],
        position_limit_m=0.08,
        rotation_limit_rad=0.2,
    )
    assert projected[:3] == pytest.approx([0.01, 0.02, 0.03])
    assert pose_delta_norms([0] * 6, projected)[1] == pytest.approx(0.2)
    assert position_limited is False
    assert rotation_limited is True


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ({"err": ["0"]}, False),
        ({"err": ["1"]}, True),
        ({"status": 0}, False),
        ({"status": "fault"}, True),
        ({}, False),
    ],
)
def test_arm_error_detection(error, expected):
    assert arm_error_active(error) is expected
