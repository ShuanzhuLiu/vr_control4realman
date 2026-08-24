from pathlib import Path


LAUNCH_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "launch"
    / "realman_tamer_left_position_live_check.launch.py"
).read_text(encoding="utf-8")


def test_position_live_check_uses_patent_canfd_path_without_optimizer():
    assert '"dry_run": False' in LAUNCH_SOURCE
    assert '"follow": True' in LAUNCH_SOURCE
    assert '"trajectory_mode": 0' in LAUNCH_SOURCE
    assert '"radio": 0' in LAUNCH_SOURCE
    assert '"canfd_pose_format": "quaternion"' in LAUNCH_SOURCE
    assert '"orientation_control_enabled": True' in LAUNCH_SOURCE
    assert 'executable="realman_pose_converter"' in LAUNCH_SOURCE
    assert '"translation_reference_frame": "controller_anchor_yaw"' in LAUNCH_SOURCE
    assert '"controller_forward_axis": "z"' in LAUNCH_SOURCE
    assert 'executable="realman_stable_pose_converter"' not in LAUNCH_SOURCE
    assert 'executable="pose_interpolator"' not in LAUNCH_SOURCE
    assert "canfd_control_space" not in LAUNCH_SOURCE
    assert "KMPPI" not in LAUNCH_SOURCE
    assert "Ruckig" not in LAUNCH_SOURCE


def test_position_live_check_matches_verified_patent_canfd_parameters():
    assert '"position_mapping_scale": 0.40' in LAUNCH_SOURCE
    assert '"rotation_mapping_scale": 0.05' in LAUNCH_SOURCE
    assert '"workspace_limit_m": 0.04' in LAUNCH_SOURCE
    assert '"rotation_workspace_limit_rad": 0.25' in LAUNCH_SOURCE
    assert '"pose_step_limit_enabled": False' in LAUNCH_SOURCE
    assert '"pose_acceleration_limit_enabled": True' in LAUNCH_SOURCE
    assert '"max_linear_velocity_mps": 0.15' in LAUNCH_SOURCE
    assert '"max_linear_acceleration_mps2": 1.2' in LAUNCH_SOURCE
    assert '"watchdog_seconds": 0.15' in LAUNCH_SOURCE
    assert 'DeclareLaunchArgument("live_confirmation", default_value="")' in LAUNCH_SOURCE
