import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from realman_teleop.pose_converter import (
    map_relative_pose,
    parse_axis_map,
    pico_euler_to_rotation,
    position_delta_in_reference_frame,
)


def test_default_axis_map_is_orthogonal_rotation():
    matrix = parse_axis_map("z,y,-x")
    assert np.linalg.det(matrix) == pytest.approx(1.0)
    assert matrix.tolist() == [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]


def test_default_axis_map_maps_controller_directions_intuitively():
    matrix = parse_axis_map("z,y,-x")
    controller_right = np.array([1.0, 0.0, 0.0])
    controller_up = np.array([0.0, 1.0, 0.0])
    controller_forward = np.array([0.0, 0.0, 1.0])
    assert matrix @ controller_forward == pytest.approx([1.0, 0.0, 0.0])
    assert matrix @ controller_up == pytest.approx([0.0, 1.0, 0.0])
    assert matrix @ controller_right == pytest.approx([0.0, 0.0, -1.0])


def test_position_and_orientation_use_same_axis_transform():
    matrix = parse_axis_map("z,y,-x")
    anchor_rotation = Rotation.identity()
    controller_rotation = Rotation.from_rotvec([0.1, 0.0, 0.0])
    target_position, target_rotation = map_relative_pose(
        [0.0, 0.0, 0.0],
        anchor_rotation,
        [0.0, 0.0, 0.0],
        Rotation.identity(),
        [0.1, 0.2, 0.3],
        controller_rotation,
        matrix,
        matrix,
    )
    assert target_position == pytest.approx([0.3, 0.2, -0.1])
    mapped_axis = target_rotation.as_rotvec()
    assert mapped_axis == pytest.approx([0.0, 0.0, -0.1])


def test_rotation_axis_map_can_flip_rotation_without_changing_position():
    position_matrix = parse_axis_map("z,y,-x")
    rotation_matrix = parse_axis_map("-z,y,x")
    target_position, target_rotation = map_relative_pose(
        [0.0, 0.0, 0.0],
        Rotation.identity(),
        [0.0, 0.0, 0.0],
        Rotation.identity(),
        [0.1, 0.2, 0.3],
        Rotation.from_rotvec([0.1, 0.0, 0.0]),
        position_matrix,
        rotation_matrix,
    )
    assert target_position == pytest.approx([0.3, 0.2, -0.1])
    assert target_rotation.as_rotvec() == pytest.approx([0.0, 0.0, 0.1])


def test_calibrated_dual_arm_position_maps_are_mirrored():
    left_matrix = parse_axis_map("z,-y,-x")
    right_matrix = parse_axis_map("z,y,x")
    controller_forward = np.array([0.0, 0.0, 1.0])
    controller_up = np.array([0.0, 1.0, 0.0])
    controller_right = np.array([1.0, 0.0, 0.0])

    assert left_matrix @ controller_forward == pytest.approx([1.0, 0.0, 0.0])
    assert right_matrix @ controller_forward == pytest.approx([1.0, 0.0, 0.0])
    assert left_matrix @ controller_up == pytest.approx([0.0, -1.0, 0.0])
    assert right_matrix @ controller_up == pytest.approx([0.0, 1.0, 0.0])
    assert left_matrix @ controller_right == pytest.approx([0.0, 0.0, -1.0])
    assert right_matrix @ controller_right == pytest.approx([0.0, 0.0, 1.0])


def test_left_pitch_sign_is_opposite_right_pitch_sign():
    left_rotation_matrix = parse_axis_map("-z,y,-x")
    right_rotation_matrix = parse_axis_map("-z,y,x")
    controller_pitch = np.array([0.1, 0.0, 0.0])

    assert left_rotation_matrix @ controller_pitch == pytest.approx([0.0, 0.0, -0.1])
    assert right_rotation_matrix @ controller_pitch == pytest.approx([0.0, 0.0, 0.1])


def test_duplicate_axis_map_is_rejected():
    with pytest.raises(ValueError, match="orthogonal"):
        parse_axis_map("z,z,-x")


def test_anchor_yaw_makes_local_forward_independent_of_facing_direction():
    facing_forward = Rotation.identity()
    facing_backward = Rotation.from_euler("y", math.pi)
    forward_delta = position_delta_in_reference_frame(
        [0.0, 0.0, 1.0], facing_forward, "controller_anchor_yaw", "z"
    )
    backward_world_delta = position_delta_in_reference_frame(
        [0.0, 0.0, -1.0], facing_backward, "controller_anchor_yaw", "z"
    )
    assert forward_delta == pytest.approx([0.0, 0.0, 1.0])
    assert backward_world_delta == pytest.approx([0.0, 0.0, 1.0])


@pytest.mark.parametrize("yaw", [-math.pi / 2, math.pi / 2])
def test_anchor_yaw_handles_controller_facing_left_or_right(yaw):
    orientation = Rotation.from_euler("y", yaw)
    world_forward = orientation.inv().apply([0.0, 0.0, 1.0])
    local_delta = position_delta_in_reference_frame(
        world_forward, orientation, "controller_anchor_yaw", "z"
    )
    assert local_delta == pytest.approx([0.0, 0.0, 1.0], abs=1e-8)


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, math.pi, -math.pi / 2])
@pytest.mark.parametrize(
    ("position_axis_map", "expected_forward"),
    [
        ("z,-y,-x", [100.0, 0.0, 0.0]),
        ("z,y,x", [100.0, 0.0, 0.0]),
    ],
)
def test_heading_compensation_preserves_dual_arm_forward_mapping(
    yaw, position_axis_map, expected_forward
):
    controller_anchor_rotation = Rotation.from_euler("y", yaw)
    tracking_forward_delta = controller_anchor_rotation.inv().apply(
        [0.0, 0.0, 100.0]
    )
    target_position, _ = map_relative_pose(
        [0.0, 0.0, 0.0],
        Rotation.identity(),
        [0.0, 0.0, 0.0],
        controller_anchor_rotation,
        tracking_forward_delta,
        controller_anchor_rotation,
        parse_axis_map(position_axis_map),
        parse_axis_map("-z,y,-x"),
        "controller_anchor_yaw",
        "z",
    )
    assert target_position == pytest.approx(expected_forward, abs=1e-8)


@pytest.mark.parametrize("pico_yaw_degrees", [-90.0, 90.0])
def test_pico_quarter_turn_yaw_maps_physical_forward_to_local_forward(
    pico_yaw_degrees,
):
    controller_rotation = pico_euler_to_rotation(
        [0.0, pico_yaw_degrees, 0.0]
    )
    physical_tracking_forward = Rotation.from_euler(
        "y", math.radians(pico_yaw_degrees)
    ).apply([0.0, 0.0, 1.0])
    local_delta = position_delta_in_reference_frame(
        physical_tracking_forward,
        controller_rotation,
        "controller_anchor_yaw",
        "z",
    )
    assert local_delta == pytest.approx([0.0, 0.0, 1.0], abs=1e-8)


def test_tracking_frame_keeps_room_direction_fixed():
    facing_backward = Rotation.from_euler("y", math.pi)
    delta = position_delta_in_reference_frame(
        [0.0, 0.0, -1.0], facing_backward, "tracking", "z"
    )
    assert delta == pytest.approx([0.0, 0.0, -1.0])


def test_invalid_controller_forward_axis_is_rejected():
    with pytest.raises(ValueError, match="controller_forward_axis"):
        position_delta_in_reference_frame(
            [0.0, 0.0, 1.0], Rotation.identity(), "controller_anchor_yaw", "y"
        )


def test_local_rotation_direction_is_independent_of_controller_facing():
    position_matrix = parse_axis_map("z,y,-x")
    rotation_matrix = parse_axis_map("-z,y,x")
    local_rotation = Rotation.from_rotvec([0.0, 0.0, 0.1])

    _, facing_forward_target = map_relative_pose(
        [0.0, 0.0, 0.0],
        Rotation.identity(),
        [0.0, 0.0, 0.0],
        Rotation.identity(),
        [0.0, 0.0, 0.0],
        local_rotation,
        position_matrix,
        rotation_matrix,
    )

    facing_backward = Rotation.from_euler("y", math.pi)
    _, facing_backward_target = map_relative_pose(
        [0.0, 0.0, 0.0],
        Rotation.identity(),
        [0.0, 0.0, 0.0],
        facing_backward,
        [0.0, 0.0, 0.0],
        facing_backward * local_rotation,
        position_matrix,
        rotation_matrix,
    )

    assert facing_forward_target.as_rotvec() == pytest.approx([-0.1, 0.0, 0.0])
    assert facing_backward_target.as_rotvec() == pytest.approx([-0.1, 0.0, 0.0])
