from pathlib import Path


LAUNCH_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "launch"
    / "realman_tamer_dual_position_live_check.launch.py"
).read_text(encoding="utf-8")


def test_dual_live_check_uses_heading_compensated_mirrored_maps():
    assert '"translation_reference_frame": "controller_anchor_yaw"' in LAUNCH_SOURCE
    assert '"controller_forward_axis": "z"' in LAUNCH_SOURCE
    assert '_converter("left", "z,-y,-x", "-z,y,-x")' in LAUNCH_SOURCE
    assert '_converter("right", "z,y,x", "-z,y,x")' in LAUNCH_SOURCE


def test_dual_live_check_isolates_arm_sdk_processes_and_controls():
    assert 'name=f"realman_{arm_name}_arm_bridge"' in LAUNCH_SOURCE
    assert '"managed_arms": arm_name' in LAUNCH_SOURCE
    assert '"left_control_enabled": arm_name == "left"' in LAUNCH_SOURCE
    assert '"right_control_enabled": arm_name == "right"' in LAUNCH_SOURCE
    assert '"169.254.128.18"' in LAUNCH_SOURCE
    assert '"169.254.128.19"' in LAUNCH_SOURCE
    assert 'f"/{arm_name}_arm/emergency_stop"' in LAUNCH_SOURCE


def test_dual_live_check_copies_verified_acceleration_limited_canfd_path():
    assert '"dry_run": False' in LAUNCH_SOURCE
    assert '"follow": True' in LAUNCH_SOURCE
    assert '"trajectory_mode": 0' in LAUNCH_SOURCE
    assert '"radio": 0' in LAUNCH_SOURCE
    assert '"canfd_pose_format": "quaternion"' in LAUNCH_SOURCE
    assert '"pose_step_limit_enabled": False' in LAUNCH_SOURCE
    assert '"pose_acceleration_limit_enabled": True' in LAUNCH_SOURCE
    assert '"max_linear_velocity_mps": 0.15' in LAUNCH_SOURCE
    assert '"max_linear_acceleration_mps2": 1.2' in LAUNCH_SOURCE
    assert '"position_mapping_scale": 0.40' in LAUNCH_SOURCE
    assert '"rotation_mapping_scale": 0.05' in LAUNCH_SOURCE
    assert '"watchdog_seconds": 0.35' in LAUNCH_SOURCE
    assert '"slow_stop_on_trigger_release": True' in LAUNCH_SOURCE
