import math
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Dict, List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation
from std_msgs.msg import Empty, String
from std_srvs.srv import SetBool

from realman_teleop.realman_arm_api_api2 import RealmanArmClient
from realman_teleop.constrained_projection import project_twist_with_joint_limits
from realman_teleop.kinematics import ModifiedDhKinematics
from realman_teleop.native_safety import RealmanNativeSafetyAnalyzer
from realman_teleop.safe_projection import (
    apply_pose_delta_twist,
    pose_delta_twist,
    project_twist_to_safe_subspace,
)
from realman_teleop.safety import (
    acceleration_limit_pose,
    arm_error_active,
    pose_delta_norms,
    project_pose_to_workspace,
    realman_canfd_pose,
    select_next_pose,
    select_pose_target,
    tamen_mm_pose_to_realman,
)


def parse_managed_arms(value: str) -> List[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not names or any(name not in ("left", "right") for name in names):
        raise ValueError("managed_arms must contain left, right, or left,right")
    if len(set(names)) != len(names):
        raise ValueError("managed_arms must not contain duplicates")
    return names


def parse_cpu_affinity(value: str, cpu_count: Optional[int] = None) -> List[int]:
    if not value.strip():
        return []
    available_cpu_count = cpu_count if cpu_count is not None else os.cpu_count()
    cpus = []
    for item in value.split(","):
        try:
            cpu = int(item.strip())
        except ValueError as exc:
            raise ValueError("cpu_affinity must be a comma-separated CPU list") from exc
        if cpu < 0 or (
            available_cpu_count is not None and cpu >= available_cpu_count
        ):
            raise ValueError(f"cpu_affinity CPU {cpu} is unavailable")
        cpus.append(cpu)
    if len(set(cpus)) != len(cpus):
        raise ValueError("cpu_affinity must not contain duplicates")
    return cpus


def evaluate_send_gap(
    interval_seconds: float,
    soft_limit_seconds: float,
    hard_limit_seconds: float,
    previous_violations: int,
    max_consecutive_violations: int,
):
    if interval_seconds <= soft_limit_seconds:
        return 0, False
    violations = previous_violations + 1
    should_fault = (
        interval_seconds >= hard_limit_seconds
        or violations >= max_consecutive_violations
    )
    return violations, should_fault


def initial_target_guard(
    elapsed_seconds: float,
    settle_seconds: float,
    position_jump_m: float,
    rotation_jump_rad: float,
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
):
    position_unsafe = position_jump_m > position_tolerance_m
    rotation_unsafe = rotation_jump_rad > rotation_tolerance_rad
    if not position_unsafe and not rotation_unsafe:
        return "accept"
    if elapsed_seconds <= settle_seconds:
        return "wait"
    return "position" if position_unsafe else "rotation"


def summarize_input_continuity(arrival_times, position_steps, rotation_steps):
    rate_hz = 0.0
    if len(arrival_times) >= 2:
        duration = arrival_times[-1] - arrival_times[0]
        if duration > 0.0:
            rate_hz = (len(arrival_times) - 1) / duration
    return (
        rate_hz,
        max(position_steps, default=0.0),
        max(rotation_steps, default=0.0),
    )


def requested_hold_mode(position_pressed: bool, orientation_pressed: bool):
    if position_pressed and orientation_pressed:
        return "combined"
    if position_pressed:
        return "position"
    if orientation_pressed:
        return "orientation"
    return None


@dataclass
class ArmRuntime:
    name: str
    client: RealmanArmClient
    enabled: bool = False
    enabled_at: float = 0.0
    anchor_pose: Optional[List[float]] = None
    last_sent_pose: Optional[List[float]] = None
    latest_target: Optional[List[float]] = None
    last_command_at: Optional[float] = None
    initial_target_accepted: bool = False
    position_pressed: bool = False
    orientation_pressed: bool = False
    active_mode: Optional[str] = None
    pending_mode: Optional[str] = None
    enable_pending: bool = False
    motion_active: bool = False
    fault_latched: bool = False
    fault_reason: str = ""
    last_send_at: Optional[float] = None
    last_send_duration_seconds: float = 0.0
    send_gap_violation_count: int = 0
    limiter_linear_velocity: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    limiter_angular_velocity: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    current_joints_rad: Optional[List[float]] = None
    last_input_target: Optional[List[float]] = None
    input_arrival_times: deque = field(
        default_factory=lambda: deque(maxlen=120), repr=False
    )
    input_position_steps: deque = field(
        default_factory=lambda: deque(maxlen=120), repr=False
    )
    input_rotation_steps: deque = field(
        default_factory=lambda: deque(maxlen=120), repr=False
    )
    state_lock: Lock = field(default_factory=Lock, repr=False)
    send_lock: Lock = field(default_factory=Lock, repr=False)


class RealmanDualArmBridge(Node):
    LIVE_CONFIRMATION = "I_UNDERSTAND_REALMAN_LIVE_CONTROL"

    def __init__(self):
        super().__init__("realman_dual_arm_bridge")
        self.dry_run = bool(self.declare_parameter("dry_run", True).value)
        self.left_ip = str(self.declare_parameter("left_ip", "169.254.128.18").value)
        self.right_ip = str(self.declare_parameter("right_ip", "169.254.128.19").value)
        self.managed_arm_names = parse_managed_arms(
            str(self.declare_parameter("managed_arms", "left,right").value)
        )
        self.robot_port = int(self.declare_parameter("robot_port", 8080).value)
        self.robot_model = str(self.declare_parameter("robot_model", "RM65").value)
        self.robot_dof = int(self.declare_parameter("robot_dof", 6).value)
        self.sdk_thread_mode = str(
            self.declare_parameter("sdk_thread_mode", "dual").value
        ).lower()
        self.cpu_affinity = parse_cpu_affinity(
            str(self.declare_parameter("cpu_affinity", "").value)
        )
        self.command_rate_hz = float(self.declare_parameter("command_rate_hz", 50.0).value)
        self.state_rate_hz = float(self.declare_parameter("state_rate_hz", 10.0).value)
        self.watchdog_seconds = float(self.declare_parameter("watchdog_seconds", 0.35).value)
        self.enable_grace_seconds = float(self.declare_parameter("enable_grace_seconds", 2.0).value)
        self.workspace_limit_m = float(self.declare_parameter("workspace_limit_m", 0.15).value)
        self.rotation_workspace_limit_rad = float(
            self.declare_parameter("rotation_workspace_limit_rad", 0.7).value
        )
        self.max_position_step_m = float(self.declare_parameter("max_position_step_m", 0.002).value)
        self.max_rotation_step_rad = float(
            self.declare_parameter("max_rotation_step_rad", 0.02).value
        )
        self.direct_target_passthrough = bool(
            self.declare_parameter("direct_target_passthrough", False).value
        )
        self.pose_step_limit_enabled = bool(
            self.declare_parameter("pose_step_limit_enabled", True).value
        )
        self.pose_acceleration_limit_enabled = bool(
            self.declare_parameter("pose_acceleration_limit_enabled", False).value
        )
        self.max_linear_velocity_mps = float(
            self.declare_parameter("max_linear_velocity_mps", 0.15).value
        )
        self.max_linear_acceleration_mps2 = float(
            self.declare_parameter("max_linear_acceleration_mps2", 1.2).value
        )
        self.max_angular_velocity_radps = float(
            self.declare_parameter("max_angular_velocity_radps", 0.5).value
        )
        self.max_angular_acceleration_radps2 = float(
            self.declare_parameter("max_angular_acceleration_radps2", 4.0).value
        )
        self.initial_position_tolerance_m = float(
            self.declare_parameter("initial_position_tolerance_m", 0.03).value
        )
        self.initial_rotation_tolerance_rad = float(
            self.declare_parameter("initial_rotation_tolerance_rad", 0.35).value
        )
        self.initial_target_settle_seconds = float(
            self.declare_parameter("initial_target_settle_seconds", 0.5).value
        )
        self.follow = bool(self.declare_parameter("follow", False).value)
        self.trajectory_mode = int(self.declare_parameter("trajectory_mode", 0).value)
        self.radio = int(self.declare_parameter("radio", 0).value)
        self.canfd_pose_format = str(
            self.declare_parameter("canfd_pose_format", "quaternion").value
        ).lower()
        self.max_send_interval_seconds = float(
            self.declare_parameter("max_send_interval_seconds", 0.02).value
        )
        self.hard_send_interval_seconds = float(
            self.declare_parameter("hard_send_interval_seconds", 0.05).value
        )
        self.max_consecutive_send_gap_violations = int(
            self.declare_parameter("max_consecutive_send_gap_violations", 3).value
        )
        self.independent_canfd_thread = bool(
            self.declare_parameter("independent_canfd_thread", False).value
        )
        self.dry_run_log_interval_seconds = float(
            self.declare_parameter("dry_run_log_interval_seconds", 2.0).value
        )
        self.trigger_control_enabled = bool(
            self.declare_parameter("trigger_control_enabled", True).value
        )
        self.trigger_mode = str(self.declare_parameter("trigger_mode", "hold").value).lower()
        self.slow_stop_on_trigger_release = bool(
            self.declare_parameter("slow_stop_on_trigger_release", False).value
        )
        self.left_control_enabled = bool(
            self.declare_parameter("left_control_enabled", True).value
        )
        self.right_control_enabled = bool(
            self.declare_parameter("right_control_enabled", True).value
        )
        self.live_confirmation = str(self.declare_parameter("live_confirmation", "").value)
        self.emergency_stop_service = str(
            self.declare_parameter("emergency_stop_service", "/realman/emergency_stop").value
        )
        self.orientation_control_enabled = bool(
            self.declare_parameter("orientation_control_enabled", True).value
        )
        self.position_mapping_scale = float(
            self.declare_parameter("position_mapping_scale", 1.0).value
        )
        self.rotation_mapping_scale = float(
            self.declare_parameter("rotation_mapping_scale", 1.0).value
        )
        self.singularity_projection_enabled = bool(
            self.declare_parameter("singularity_projection_enabled", False).value
        )
        self.projection_hard_threshold = float(
            self.declare_parameter("projection_hard_threshold", 0.02).value
        )
        self.projection_soft_threshold = float(
            self.declare_parameter("projection_soft_threshold", 0.08).value
        )
        self.projection_characteristic_length_m = float(
            self.declare_parameter("projection_characteristic_length_m", 0.3).value
        )
        self.fault_on_native_safety = bool(
            self.declare_parameter("fault_on_native_safety", False).value
        )
        self.joint_limit_projection_enabled = bool(
            self.declare_parameter("joint_limit_projection_enabled", False).value
        )
        self.joint_limit_margin_rad = float(
            self.declare_parameter("joint_limit_margin_rad", 0.05).value
        )
        self.joint_limit_prediction_horizon_s = float(
            self.declare_parameter("joint_limit_prediction_horizon_s", 0.25).value
        )
        self.joint_limit_projection_damping = float(
            self.declare_parameter("joint_limit_projection_damping", 0.02).value
        )
        self.control_allowed = {
            "left": self.left_control_enabled and "left" in self.managed_arm_names,
            "right": self.right_control_enabled and "right" in self.managed_arm_names,
        }

        self._validate_parameters()
        self._apply_cpu_affinity()
        self._last_log_times: Dict[str, float] = {}
        self._closing = False
        self.arms: Dict[str, ArmRuntime] = {}
        self.joint_publishers = {}
        self.legacy_tcp_publishers = {}
        self.pose_publishers = {}
        self.converter_clients = {}
        self.interpolator_reset_publishers = {}
        self.kinematics_models = {}
        self.native_safety_analyzers = {}
        self.safety_metrics_publishers = {}
        self._sender_stop = Event()
        self._sender_threads: List[Thread] = []

        try:
            arm_ips = {"left": self.left_ip, "right": self.right_ip}
            for arm_name in self.managed_arm_names:
                self._create_arm(arm_name, arm_ips[arm_name])
        except Exception:
            self._disconnect_all()
            raise

        for arm_name in self.arms:
            self.create_subscription(
                JointState,
                f"/{arm_name}_arm/interpolated_robot_command",
                lambda message, name=arm_name: self._command_callback(name, message),
                1,
            )
            self.joint_publishers[arm_name] = self.create_publisher(
                JointState, f"/{arm_name}_arm/joint_states", 10
            )
            self.legacy_tcp_publishers[arm_name] = self.create_publisher(
                JointState, f"/{arm_name}_arm/cur_tcp_pose", 10
            )
            self.pose_publishers[arm_name] = self.create_publisher(
                PoseStamped, f"/{arm_name}_arm/tcp_pose", 10
            )
            self.safety_metrics_publishers[arm_name] = self.create_publisher(
                String, f"/{arm_name}_arm/safety_metrics", 10
            )
            self.create_service(
                SetBool,
                f"/{arm_name}_arm/servo_move_enable",
                lambda request, response, name=arm_name: self._enable_callback(
                    name, request, response
                ),
            )
            self.create_service(
                SetBool,
                f"/{arm_name}_arm/servo_position_enable",
                lambda request, response, name=arm_name: self._mode_enable_callback(
                    name, "position", request, response
                ),
            )
            self.create_service(
                SetBool,
                f"/{arm_name}_arm/servo_orientation_enable",
                lambda request, response, name=arm_name: self._mode_enable_callback(
                    name, "orientation", request, response
                ),
            )
            self.create_service(
                SetBool,
                f"/{arm_name}_arm/clear_fault",
                lambda request, response, name=arm_name: self._clear_fault_callback(
                    name, request, response
                ),
            )
            self.converter_clients[arm_name] = self.create_client(
                SetBool, f"/{arm_name}_arm/vr_robot_pose_converter"
            )
            self.interpolator_reset_publishers[arm_name] = self.create_publisher(
                Empty, f"/{arm_name}_arm/pose_interpolator_reset", 10
            )

        if self.trigger_control_enabled and "left" in self.arms:
            self.create_subscription(
                String,
                "/vr_left_front_button",
                lambda message: self._mode_button_callback("left", "position", "LT", message),
                10,
            )
            self.create_subscription(
                String,
                "/vr_left_side_button",
                lambda message: self._mode_button_callback("left", "orientation", "LG", message),
                10,
            )
        if self.trigger_control_enabled and "right" in self.arms:
            self.create_subscription(
                String,
                "/vr_right_front_button",
                lambda message: self._mode_button_callback("right", "position", "RT", message),
                10,
            )
            self.create_subscription(
                String,
                "/vr_right_side_button",
                lambda message: self._mode_button_callback("right", "orientation", "RG", message),
                10,
            )

        self.create_service(
            SetBool, self.emergency_stop_service, self._emergency_stop_callback
        )
        self.command_timer = None
        if self.independent_canfd_thread:
            for runtime in self.arms.values():
                thread = Thread(
                    target=self._sender_loop,
                    args=(runtime,),
                    daemon=True,
                    name=f"realman-canfd-{runtime.name}",
                )
                thread.start()
                self._sender_threads.append(thread)
        else:
            self.command_timer = self.create_timer(1.0 / self.command_rate_hz, self._command_timer)
        self.state_timer = self.create_timer(1.0 / self.state_rate_hz, self._state_timer)

        mode = "DRY-RUN" if self.dry_run else "LIVE"
        allowed = ",".join(name for name, enabled in self.control_allowed.items() if enabled)
        if self.direct_target_passthrough:
            target_path = "direct-passthrough"
        elif self.pose_acceleration_limit_enabled:
            target_path = (
                f"scaled-accel-limited({self.position_mapping_scale:.3f}/"
                f"{self.rotation_mapping_scale:.3f})"
            )
        elif self.pose_step_limit_enabled:
            target_path = (
                f"scaled-step-limited({self.position_mapping_scale:.3f}/"
                f"{self.rotation_mapping_scale:.3f})"
            )
        else:
            target_path = (
                f"scaled-passthrough({self.position_mapping_scale:.3f}/"
                f"{self.rotation_mapping_scale:.3f})"
            )
        rate_limit_text = ""
        if self.pose_acceleration_limit_enabled:
            rate_limit_text = (
                f", cartesian_limits={self.max_linear_velocity_mps * 1000.0:.0f}mm/s/"
                f"{self.max_linear_acceleration_mps2 * 1000.0:.0f}mm/s2, "
                f"angular_limits={self.max_angular_velocity_radps:.2f}rad/s/"
                f"{self.max_angular_acceleration_radps2:.2f}rad/s2"
            )
        self.get_logger().warn(
            f"RealMan dual-arm bridge started in {mode} mode; follow={self.follow}, "
            f"command_rate={self.command_rate_hz:.1f}Hz, trigger_mode={self.trigger_mode}, "
            f"control={allowed or 'none'}, orientation={self.orientation_control_enabled}, "
            f"managed={','.join(self.managed_arm_names)}, sdk={self.sdk_thread_mode}, "
            f"cpu={','.join(str(cpu) for cpu in self.cpu_affinity) or 'any'}, "
            f"target_path={target_path}, "
            f"canfd_pose={self.canfd_pose_format}, "
            f"trajectory={self.trajectory_mode}/{self.radio}, "
            f"trigger_release_stop="
            f"{'slow' if self.slow_stop_on_trigger_release else 'hard'}, "
            f"sender={'thread' if self.independent_canfd_thread else 'ros-timer'}, "
            f"max_send_gap={self.max_send_interval_seconds * 1000.0:.1f}ms, "
            f"hard_gap={self.hard_send_interval_seconds * 1000.0:.1f}ms, "
            f"projection={self.singularity_projection_enabled}/"
            f"{self.joint_limit_projection_enabled}"
            f"{rate_limit_text}"
        )
        if self.follow and self.max_send_interval_seconds > 0.010:
            self.get_logger().warn(
                "High-follow send-gap threshold exceeds the SDK's 10ms recommendation; "
                "use only for controlled testing"
            )

    def _validate_parameters(self) -> None:
        positive_values = {
            "command_rate_hz": self.command_rate_hz,
            "state_rate_hz": self.state_rate_hz,
            "watchdog_seconds": self.watchdog_seconds,
            "enable_grace_seconds": self.enable_grace_seconds,
            "workspace_limit_m": self.workspace_limit_m,
            "rotation_workspace_limit_rad": self.rotation_workspace_limit_rad,
            "initial_position_tolerance_m": self.initial_position_tolerance_m,
            "initial_rotation_tolerance_rad": self.initial_rotation_tolerance_rad,
            "initial_target_settle_seconds": self.initial_target_settle_seconds,
            "dry_run_log_interval_seconds": self.dry_run_log_interval_seconds,
            "max_send_interval_seconds": self.max_send_interval_seconds,
            "hard_send_interval_seconds": self.hard_send_interval_seconds,
            "projection_hard_threshold": self.projection_hard_threshold,
            "projection_soft_threshold": self.projection_soft_threshold,
            "projection_characteristic_length_m": self.projection_characteristic_length_m,
            "joint_limit_prediction_horizon_s": self.joint_limit_prediction_horizon_s,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.pose_step_limit_enabled and not self.direct_target_passthrough:
            for name, value in {
                "max_position_step_m": self.max_position_step_m,
                "max_rotation_step_rad": self.max_rotation_step_rad,
            }.items():
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
        if self.pose_acceleration_limit_enabled and not self.direct_target_passthrough:
            for name, value in {
                "max_linear_velocity_mps": self.max_linear_velocity_mps,
                "max_linear_acceleration_mps2": self.max_linear_acceleration_mps2,
                "max_angular_velocity_radps": self.max_angular_velocity_radps,
                "max_angular_acceleration_radps2": self.max_angular_acceleration_radps2,
            }.items():
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
        if self.pose_step_limit_enabled and self.pose_acceleration_limit_enabled:
            raise ValueError(
                "pose_step_limit_enabled and pose_acceleration_limit_enabled are mutually exclusive"
            )
        if not self.direct_target_passthrough:
            for name, value in {
                "position_mapping_scale": self.position_mapping_scale,
                "rotation_mapping_scale": self.rotation_mapping_scale,
            }.items():
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and non-negative")
            if self.position_mapping_scale == 0 and self.rotation_mapping_scale == 0:
                raise ValueError("at least one mapping scale must be greater than zero")
        if self.projection_soft_threshold <= self.projection_hard_threshold:
            raise ValueError("projection thresholds must satisfy hard < soft")
        if self.joint_limit_margin_rad < 0 or self.joint_limit_projection_damping < 0:
            raise ValueError("joint-limit margin and damping must be non-negative")
        if (
            self.singularity_projection_enabled or self.joint_limit_projection_enabled
        ) and not self.dry_run:
            raise ValueError("safety projection is currently restricted to dry-run experiments")
        if self.follow and self.command_rate_hz < 100.0:
            raise ValueError("follow=true requires command_rate_hz >= 100")
        if self.follow and self.max_send_interval_seconds <= 1.0 / self.command_rate_hz:
            raise ValueError(
                "max_send_interval_seconds must exceed the nominal command period"
            )
        if self.hard_send_interval_seconds <= self.max_send_interval_seconds:
            raise ValueError(
                "hard_send_interval_seconds must exceed max_send_interval_seconds"
            )
        if self.max_consecutive_send_gap_violations < 1:
            raise ValueError("max_consecutive_send_gap_violations must be positive")
        if self.trajectory_mode not in (0, 1, 2):
            raise ValueError("trajectory_mode must be 0, 1, or 2")
        if self.canfd_pose_format not in ("euler", "quaternion"):
            raise ValueError("canfd_pose_format must be euler or quaternion")
        if self.radio < 0:
            raise ValueError("radio must be non-negative")
        if self.trigger_mode not in ("hold", "toggle"):
            raise ValueError("trigger_mode must be 'hold' or 'toggle'")
        sdk_thread_mode = getattr(self, "sdk_thread_mode", "dual")
        if sdk_thread_mode not in ("single", "dual", "triple"):
            raise ValueError("sdk_thread_mode must be single, dual, or triple")
        emergency_stop_service = getattr(
            self, "emergency_stop_service", "/realman/emergency_stop"
        )
        if not emergency_stop_service.startswith("/"):
            raise ValueError("emergency_stop_service must be an absolute ROS service name")
        if not self.dry_run and self.live_confirmation != self.LIVE_CONFIRMATION:
            raise ValueError(
                "live mode requires live_confirmation=" + self.LIVE_CONFIRMATION
            )

    def _apply_cpu_affinity(self) -> None:
        if not self.cpu_affinity:
            return
        try:
            os.sched_setaffinity(0, set(self.cpu_affinity))
        except (AttributeError, OSError) as exc:
            self.get_logger().warn(f"CPU affinity not applied: {exc}")

    def _create_arm(self, name: str, ip: str) -> None:
        client = RealmanArmClient(
            ip=ip,
            port=self.robot_port,
            model=self.robot_model,
            dof=self.robot_dof,
            thread_mode=self.sdk_thread_mode,
        )
        client.connect()
        state = client.get_state()
        if arm_error_active(state.err):
            client.disconnect()
            raise RuntimeError(f"{name} arm reports an error: {state.err}")
        self.kinematics_models[name] = ModifiedDhKinematics.from_client(client)
        self.native_safety_analyzers[name] = RealmanNativeSafetyAnalyzer(client)
        self.arms[name] = ArmRuntime(
            name=name,
            client=client,
            current_joints_rad=[math.radians(value) for value in state.joints],
        )
        self.get_logger().info(
            f"Connected {name} arm at {ip}:{self.robot_port}; "
            f"power={client.get_power_state()}, pose={state.pose.as_list()}"
        )

    def _command_callback(self, arm_name: str, message: JointState) -> None:
        runtime = self.arms[arm_name]
        try:
            target = tamen_mm_pose_to_realman(message.position)
            received_at = time.monotonic()
            with runtime.state_lock:
                if not runtime.enabled:
                    return
                if runtime.last_input_target is not None:
                    position_step, rotation_step = pose_delta_norms(
                        runtime.last_input_target,
                        target,
                    )
                    if runtime.active_mode == "position":
                        rotation_step = 0.0
                    elif runtime.active_mode == "orientation":
                        position_step = 0.0
                    runtime.input_position_steps.append(position_step)
                    runtime.input_rotation_steps.append(rotation_step)
                runtime.last_input_target = target
                runtime.input_arrival_times.append(received_at)
                runtime.latest_target = target
                runtime.last_command_at = received_at
        except ValueError as exc:
            self._fault_arm(runtime, f"invalid command: {exc}")

    def _enable_callback(self, arm_name: str, request, response):
        runtime = self.arms[arm_name]
        if request.data:
            if not self.control_allowed[arm_name]:
                response.success = False
                response.message = f"{arm_name} arm control is disabled by configuration"
                return response
            self._reset_interpolator(arm_name)
            response.success, response.message = self._enable_runtime(runtime, "combined")
        else:
            self._disable_arm(runtime, "service request", stop=True)
            response.success = True
            response.message = f"{arm_name} arm disabled"
        return response

    def _mode_enable_callback(self, arm_name: str, mode: str, request, response):
        if mode not in ("position", "orientation"):
            response.success = False
            response.message = f"unsupported control mode: {mode}"
            return response
        if mode == "orientation" and not self.orientation_control_enabled:
            response.success = False
            response.message = "orientation control is disabled by configuration"
            return response
        runtime = self.arms[arm_name]
        if request.data:
            if not self.control_allowed[arm_name]:
                response.success = False
                response.message = f"{arm_name} arm control is disabled by configuration"
                return response
            self._reset_interpolator(arm_name)
            response.success, response.message = self._enable_runtime(runtime, mode)
        else:
            self._disable_arm(runtime, "service request", stop=True)
            response.success = True
            response.message = f"{arm_name} arm disabled"
        return response

    def _enable_runtime(self, runtime: ArmRuntime, mode: str):
        try:
            if runtime.fault_latched:
                raise RuntimeError(f"fault latched: {runtime.fault_reason}")
            state = runtime.client.get_state()
            if arm_error_active(state.err):
                raise RuntimeError(f"arm error: {state.err}")
            if not runtime.client.get_power_state():
                raise RuntimeError("arm is not powered on")
            pose = state.pose.as_list()
            joints_rad = [math.radians(value) for value in state.joints]
            with runtime.state_lock:
                runtime.enabled = True
                runtime.enabled_at = time.monotonic()
                runtime.anchor_pose = pose
                runtime.last_sent_pose = pose
                runtime.current_joints_rad = joints_rad
                runtime.latest_target = None
                runtime.last_command_at = None
                runtime.last_input_target = None
                runtime.input_arrival_times.clear()
                runtime.input_position_steps.clear()
                runtime.input_rotation_steps.clear()
                runtime.initial_target_accepted = False
                runtime.motion_active = False
                runtime.last_send_at = None
                runtime.last_send_duration_seconds = 0.0
                runtime.send_gap_violation_count = 0
                runtime.limiter_linear_velocity = [0.0, 0.0, 0.0]
                runtime.limiter_angular_velocity = [0.0, 0.0, 0.0]
                runtime.active_mode = mode
            message = (
                f"{runtime.name} arm enabled at anchor "
                f"{json.dumps(pose, separators=(',', ':'))}"
            )
            mode = "DRY" if self.dry_run else "LIVE"
            self.get_logger().warn(
                f"{runtime.name.upper()} {runtime.active_mode.upper()} enabled ({mode})"
            )
            self.get_logger().debug(f"{runtime.name} anchor pose: {pose}")
            return True, message
        except Exception as exc:
            runtime.enabled = False
            return False, str(exc)

    def _mode_button_callback(
        self, arm_name: str, control_mode: str, prefix: str, message: String
    ) -> None:
        if not self.control_allowed[arm_name]:
            return
        if control_mode == "orientation" and not self.orientation_control_enabled:
            return
        runtime = self.arms[arm_name]
        value = message.data.strip()
        if value not in (f"{prefix}=T", f"{prefix}=F"):
            return
        pressed = value.endswith("T")
        field_name = f"{control_mode}_pressed"
        with runtime.state_lock:
            if pressed == getattr(runtime, field_name):
                return
            setattr(runtime, field_name, pressed)
            conflict = runtime.position_pressed and runtime.orientation_pressed
            requested_mode = requested_hold_mode(
                runtime.position_pressed,
                runtime.orientation_pressed,
            )
            active_mode = runtime.active_mode
            pending_mode = runtime.pending_mode
            enabled = runtime.enabled
            enable_pending = runtime.enable_pending
            if self.trigger_mode == "hold" and enable_pending:
                runtime.pending_mode = requested_mode

        if self.trigger_mode == "hold":
            if enable_pending:
                if requested_mode is None:
                    self._request_trigger_disable(
                        runtime,
                        f"{control_mode} button released",
                    )
                return
            if requested_mode is None:
                if enabled:
                    self._request_trigger_disable(
                        runtime,
                        f"{control_mode} button released",
                    )
                return
            if not enabled:
                self._request_trigger_enable(runtime, requested_mode)
            elif active_mode != requested_mode:
                self._request_trigger_mode_switch(runtime, requested_mode)
            return

        if conflict:
            self._request_trigger_disable(runtime, "position/orientation conflict")
            return

        if pressed:
            if enabled and active_mode == control_mode:
                self._request_trigger_disable(runtime, f"{control_mode} toggled off")
            else:
                if enabled or enable_pending or pending_mode is not None:
                    self._request_trigger_disable(runtime, "control mode switch")
                self._request_trigger_enable(runtime, control_mode)

    def _request_trigger_enable(self, runtime: ArmRuntime, control_mode: str) -> None:
        if not self.control_allowed[runtime.name]:
            return
        with runtime.state_lock:
            fault_latched = runtime.fault_latched
            fault_reason = runtime.fault_reason
            unavailable = runtime.enabled or runtime.enable_pending
            if not fault_latched and not unavailable:
                runtime.enable_pending = True
                runtime.pending_mode = control_mode
        if fault_latched:
            self.get_logger().error(
                f"{runtime.name.upper()} blocked by latched fault: {fault_reason}"
            )
            return
        if unavailable:
            return
        self._reset_interpolator(runtime.name)
        self._request_converter(
            runtime.name,
            True,
            lambda success, message: self._complete_trigger_enable(
                runtime, control_mode, success, message
            ),
        )

    def _complete_trigger_enable(
        self, runtime: ArmRuntime, control_mode: str, success: bool, message: str
    ) -> None:
        with runtime.state_lock:
            enable_pending = runtime.enable_pending
            if self.trigger_mode == "hold":
                selected_mode = requested_hold_mode(
                    runtime.position_pressed,
                    runtime.orientation_pressed,
                )
                runtime.pending_mode = selected_mode
            else:
                selected_mode = control_mode
        if not enable_pending:
            return
        if selected_mode is None:
            with runtime.state_lock:
                runtime.enable_pending = False
                runtime.pending_mode = None
            self._request_converter(runtime.name, False)
            return
        if not success:
            with runtime.state_lock:
                runtime.enable_pending = False
                runtime.pending_mode = None
            self.get_logger().error(f"{runtime.name} converter enable failed: {message}")
            return
        with runtime.state_lock:
            runtime.enable_pending = False
            runtime.pending_mode = None
        enabled, enable_message = self._enable_runtime(runtime, selected_mode)
        if not enabled:
            self.get_logger().error(f"{runtime.name} trigger enable failed: {enable_message}")
            self._request_converter(runtime.name, False)

    def _request_trigger_mode_switch(
        self, runtime: ArmRuntime, control_mode: str
    ) -> None:
        with runtime.state_lock:
            if not runtime.enabled or runtime.active_mode == control_mode:
                return
            previous_mode = runtime.active_mode
        reason = f"control mode {previous_mode}->{control_mode}"
        self._disable_arm(
            runtime,
            reason,
            stop=True,
            slow_stop=self.slow_stop_on_trigger_release,
        )
        self._reset_interpolator(runtime.name)
        with runtime.state_lock:
            selected_mode = requested_hold_mode(
                runtime.position_pressed,
                runtime.orientation_pressed,
            )
            if selected_mode is not None:
                runtime.enable_pending = True
                runtime.pending_mode = selected_mode
        if selected_mode is None:
            self._request_converter(runtime.name, False)
            return
        self._request_converter(
            runtime.name,
            True,
            lambda success, message: self._complete_trigger_enable(
                runtime, selected_mode, success, message
            ),
        )

    def _request_trigger_disable(self, runtime: ArmRuntime, reason: str) -> None:
        with runtime.state_lock:
            runtime.enable_pending = False
            runtime.pending_mode = None
        normal_release = reason.endswith("button released") or reason.endswith(
            "toggled off"
        )
        self._disable_arm(
            runtime,
            reason,
            stop=True,
            slow_stop=self.slow_stop_on_trigger_release and normal_release,
        )
        self._request_converter(runtime.name, False)
        self._reset_interpolator(runtime.name)

    def _reset_interpolator(self, arm_name: str) -> None:
        publisher = getattr(self, "interpolator_reset_publishers", {}).get(arm_name)
        if publisher is not None:
            publisher.publish(Empty())

    def _request_converter(self, arm_name: str, enabled: bool, callback=None) -> None:
        client = self.converter_clients[arm_name]
        if not client.service_is_ready():
            message = f"/{arm_name}_arm/vr_robot_pose_converter is unavailable"
            if callback is not None:
                callback(False, message)
            else:
                self.get_logger().error(message)
            return
        request = SetBool.Request()
        request.data = enabled
        future = client.call_async(request)
        future.add_done_callback(
            lambda completed: self._converter_response(arm_name, enabled, completed, callback)
        )

    def _converter_response(self, arm_name: str, enabled: bool, future, callback) -> None:
        try:
            response = future.result()
            success = bool(response.success)
            message = response.message
        except Exception as exc:
            success = False
            message = str(exc)
        action = "enabled" if enabled else "disabled"
        if success:
            self.get_logger().debug(f"{arm_name} pose converter {action}")
        else:
            self.get_logger().error(f"{arm_name} pose converter {action} failed: {message}")
        if callback is not None:
            callback(success, message)

    def _command_timer(self) -> None:
        now = time.monotonic()
        for runtime in self.arms.values():
            self._process_arm_command(runtime, now)

    def _sender_loop(self, runtime: ArmRuntime) -> None:
        period = 1.0 / self.command_rate_hz
        next_deadline = time.monotonic()
        while not self._sender_stop.is_set():
            with runtime.state_lock:
                enabled = runtime.enabled
            if not enabled:
                self._sender_stop.wait(min(0.05, period))
                next_deadline = time.monotonic()
                continue
            now = time.monotonic()
            delay = next_deadline - now
            if delay > 0:
                self._sender_stop.wait(delay)
                continue
            self._process_arm_command(runtime, now)
            next_deadline += period
            if next_deadline < now - period:
                next_deadline = now + period

    def _process_arm_command(self, runtime: ArmRuntime, now: float) -> None:
        with runtime.state_lock:
            if not runtime.enabled:
                return
            enabled_at = runtime.enabled_at
            last_command_at = runtime.last_command_at
            raw_target = list(runtime.latest_target) if runtime.latest_target is not None else None
            anchor = list(runtime.anchor_pose) if runtime.anchor_pose is not None else None
            current = list(runtime.last_sent_pose) if runtime.last_sent_pose is not None else None
            active_mode = runtime.active_mode
            initial_target_accepted = runtime.initial_target_accepted
            current_joints_rad = (
                list(runtime.current_joints_rad)
                if runtime.current_joints_rad is not None
                else None
            )
            input_rate_hz, input_position_step_max, input_rotation_step_max = (
                summarize_input_continuity(
                    runtime.input_arrival_times,
                    runtime.input_position_steps,
                    runtime.input_rotation_steps,
                )
            )
            limiter_linear_velocity = list(runtime.limiter_linear_velocity)
            limiter_angular_velocity = list(runtime.limiter_angular_velocity)

        if raw_target is None or last_command_at is None:
            if now - enabled_at > self.enable_grace_seconds:
                self._fault_arm(runtime, "no command received during enable grace period")
            return
        if now - last_command_at > self.watchdog_seconds:
            self._fault_arm(runtime, "command watchdog timeout")
            return
        if anchor is None or current is None or active_mode is None:
            self._fault_arm(runtime, "missing enable anchor or control mode")
            return

        target = select_pose_target(
            anchor,
            raw_target,
            self.position_mapping_scale,
            self.rotation_mapping_scale,
            self.direct_target_passthrough,
        )
        if active_mode == "position":
            target[3:6] = anchor[3:6]
        elif active_mode == "orientation":
            target[:3] = anchor[:3]
        elif active_mode == "combined" and not self.orientation_control_enabled:
            target[3:6] = anchor[3:6]

        target, position_boundary, rotation_boundary = project_pose_to_workspace(
            anchor,
            target,
            self.workspace_limit_m,
            self.rotation_workspace_limit_rad,
        )
        position_from_anchor, rotation_from_anchor = pose_delta_norms(anchor, target)
        if not initial_target_accepted:
            initial_status = initial_target_guard(
                now - enabled_at,
                self.initial_target_settle_seconds,
                position_from_anchor,
                rotation_from_anchor,
                self.initial_position_tolerance_m,
                self.initial_rotation_tolerance_rad,
            )
            if initial_status == "wait":
                self._log_throttled(
                    f"initial_settle_{runtime.name}",
                    f"{runtime.name.upper()} waiting for fresh initial target: "
                    f"position={position_from_anchor:.4f}m, "
                    f"rotation={rotation_from_anchor:.4f}rad",
                    0.25,
                )
                return
            if initial_status == "position":
                self._fault_arm(
                    runtime,
                    f"initial position jump {position_from_anchor:.4f}m exceeds "
                    f"{self.initial_position_tolerance_m:.4f}m",
                )
                return
            if initial_status == "rotation":
                self._fault_arm(
                    runtime,
                    f"initial rotation jump {rotation_from_anchor:.4f}rad exceeds "
                    f"{self.initial_rotation_tolerance_rad:.4f}rad",
                )
                return
            with runtime.state_lock:
                if runtime.enabled:
                    runtime.initial_target_accepted = True

        projection_result = None
        joint_projection_result = None
        if self.singularity_projection_enabled or self.joint_limit_projection_enabled:
            if current_joints_rad is None:
                self._fault_arm(runtime, "missing joint state for singularity projection")
                return
            desired_delta = pose_delta_twist(anchor, target)
            projected_delta = desired_delta
        if self.singularity_projection_enabled:
            projection_result = project_twist_to_safe_subspace(
                self.kinematics_models[runtime.name].jacobian(current_joints_rad),
                projected_delta,
                hard_threshold=self.projection_hard_threshold,
                soft_threshold=self.projection_soft_threshold,
                characteristic_length_m=self.projection_characteristic_length_m,
            )
            projected_delta = projection_result.safe_twist
        if self.joint_limit_projection_enabled:
            model = self.kinematics_models[runtime.name]
            joint_projection_result = project_twist_with_joint_limits(
                model.jacobian(current_joints_rad),
                projected_delta,
                current_joints_rad,
                model.lower_limits,
                model.upper_limits,
                model.velocity_limits,
                position_margin_rad=self.joint_limit_margin_rad,
                prediction_horizon_s=self.joint_limit_prediction_horizon_s,
                damping=self.joint_limit_projection_damping,
                characteristic_length_m=self.projection_characteristic_length_m,
            )
            projected_delta = joint_projection_result.safe_twist
        if self.singularity_projection_enabled or self.joint_limit_projection_enabled:
            target = apply_pose_delta_twist(anchor, projected_delta)
            position_from_anchor, rotation_from_anchor = pose_delta_norms(anchor, target)

        next_linear_velocity = limiter_linear_velocity
        next_angular_velocity = limiter_angular_velocity
        if self.pose_acceleration_limit_enabled and not self.direct_target_passthrough:
            (
                next_pose,
                next_linear_velocity,
                next_angular_velocity,
                clamped,
            ) = acceleration_limit_pose(
                current,
                target,
                limiter_linear_velocity,
                limiter_angular_velocity,
                1.0 / self.command_rate_hz,
                self.max_linear_velocity_mps,
                self.max_linear_acceleration_mps2,
                self.max_angular_velocity_radps,
                self.max_angular_acceleration_radps2,
            )
        else:
            next_pose, clamped = select_next_pose(
                current,
                target,
                self.max_position_step_m,
                self.max_rotation_step_rad,
                self.direct_target_passthrough or not self.pose_step_limit_enabled,
            )
        if self.dry_run:
            delta_xyz = [(target[index] - anchor[index]) * 1000.0 for index in range(3)]
            delta_text = ",".join(f"{value:+.1f}" for value in delta_xyz)
            step_text = "RAMP" if clamped and self.pose_acceleration_limit_enabled else (
                "LIMITED" if clamped else "OK"
            )
            boundary_text = " BOUNDARY" if position_boundary or rotation_boundary else ""
            projection_text = (
                ""
                if projection_result is None
                else f" proj={projection_result.retained_ratio:.2f}"
            )
            joint_limit_text = ""
            if joint_projection_result is not None:
                active_count = int(
                    np.count_nonzero(joint_projection_result.active_lower_limits)
                    + np.count_nonzero(joint_projection_result.active_upper_limits)
                )
                joint_limit_text = (
                    f" joint={joint_projection_result.retained_ratio:.2f}"
                    f" active={active_count}"
                )
            target_path_text = (
                "path=DIRECT"
                if self.direct_target_passthrough
                else (
                    f"path=SCALED scale={self.position_mapping_scale:.2f}/"
                    f"{self.rotation_mapping_scale:.2f}"
                )
            )
            self._log_throttled(
                f"dry_{runtime.name}",
                f"{runtime.name.upper()} {active_mode.upper()} "
                f"dXYZ=[{delta_text}]mm "
                f"dR={math.degrees(rotation_from_anchor):.1f}deg step={step_text} "
                f"{target_path_text} "
                f"inputHz={input_rate_hz:.1f} "
                f"frameMaxXYZ={input_position_step_max * 1000.0:.1f}mm "
                f"frameMaxR={math.degrees(input_rotation_step_max):.1f}deg"
                f"{boundary_text}{projection_text}{joint_limit_text}",
                self.dry_run_log_interval_seconds,
            )
            with runtime.state_lock:
                if runtime.enabled:
                    runtime.last_sent_pose = next_pose
                    runtime.limiter_linear_velocity = next_linear_velocity
                    runtime.limiter_angular_velocity = next_angular_velocity
            return

        gap_error = None
        gap_warning = None
        try:
            with runtime.send_lock:
                with runtime.state_lock:
                    if not runtime.enabled:
                        return
                    motion_active = runtime.motion_active
                    last_send_at = runtime.last_send_at
                    previous_send_duration = runtime.last_send_duration_seconds
                    previous_gap_violations = runtime.send_gap_violation_count
                send_started_at = time.monotonic()
                send_interval = (
                    send_started_at - last_send_at
                    if last_send_at is not None
                    else 0.0
                )
                if (
                    self.follow
                    and motion_active
                    and last_send_at is not None
                    and send_interval > self.max_send_interval_seconds
                ):
                    gap_violations, should_fault = evaluate_send_gap(
                        send_interval,
                        self.max_send_interval_seconds,
                        self.hard_send_interval_seconds,
                        previous_gap_violations,
                        self.max_consecutive_send_gap_violations,
                    )
                    gap_message = (
                        f"high-follow send gap {send_interval:.4f}s exceeds "
                        f"{self.max_send_interval_seconds:.4f}s; "
                        f"previous_sdk_call={previous_send_duration:.4f}s, "
                        f"scheduler_gap={max(0.0, send_interval - previous_send_duration):.4f}s, "
                        f"violations={gap_violations}/"
                        f"{self.max_consecutive_send_gap_violations}"
                    )
                    with runtime.state_lock:
                        runtime.send_gap_violation_count = gap_violations
                    if should_fault:
                        gap_error = gap_message
                    else:
                        gap_warning = gap_message
                else:
                    with runtime.state_lock:
                        runtime.send_gap_violation_count = 0

                if gap_error is None:
                    sdk_call_started_at = time.monotonic()
                    runtime.client.movep_canfd(
                        realman_canfd_pose(next_pose, self.canfd_pose_format),
                        follow=self.follow,
                        trajectory_mode=self.trajectory_mode,
                        radio=self.radio,
                    )
                    sdk_call_duration = time.monotonic() - sdk_call_started_at
                    with runtime.state_lock:
                        if runtime.enabled:
                            runtime.motion_active = True
                            runtime.last_send_at = send_started_at
                            runtime.last_send_duration_seconds = sdk_call_duration
                            runtime.last_sent_pose = next_pose
                            runtime.limiter_linear_velocity = next_linear_velocity
                            runtime.limiter_angular_velocity = next_angular_velocity
                    if self.follow and sdk_call_duration >= self.hard_send_interval_seconds:
                        gap_error = (
                            f"high-follow SDK call duration {sdk_call_duration:.4f}s "
                            f"exceeds hard limit {self.hard_send_interval_seconds:.4f}s"
                        )
        except Exception as exc:
            self._fault_arm(runtime, f"movep_canfd failed: {exc}")
            return
        if gap_error is not None:
            self._fault_arm(runtime, gap_error)
        elif gap_warning is not None:
            self.get_logger().warn(f"{runtime.name.upper()} jitter: {gap_warning}")

    def _state_timer(self) -> None:
        if self._closing or not rclpy.ok():
            return
        if self.follow:
            for runtime in self.arms.values():
                with runtime.state_lock:
                    if runtime.enabled:
                        return
        for runtime in self.arms.values():
            try:
                state = runtime.client.get_state()
                with runtime.state_lock:
                    runtime.current_joints_rad = [
                        math.radians(value) for value in state.joints
                    ]
                self._publish_state(runtime.name, state)
                metrics = self.native_safety_analyzers[runtime.name].analyze(state.joints)
                model_metrics = self.kinematics_models[runtime.name].metrics(
                    [math.radians(value) for value in state.joints]
                )
                metrics_message = String()
                metrics_message.data = json.dumps(
                    {
                        "analytic_singularity": metrics.analytic_singularity_label,
                        "analytic_code": metrics.analytic_singularity_code,
                        "universal_singular": metrics.universal_singular,
                        "self_collision": metrics.self_collision,
                        "shoulder_plane_distance_m": metrics.shoulder_plane_distance_m,
                        "minimum_joint_limit_margin_deg": metrics.minimum_joint_limit_margin_deg,
                        "sigma_min": model_metrics.sigma_min,
                        "condition_number": model_metrics.condition_number,
                        "manipulability": model_metrics.manipulability,
                    },
                    separators=(",", ":"),
                )
                self.safety_metrics_publishers[runtime.name].publish(metrics_message)
                with runtime.state_lock:
                    enabled = runtime.enabled
                if enabled and self.fault_on_native_safety and metrics.unsafe:
                    self._fault_arm(runtime, f"native safety metric unsafe: {metrics_message.data}")
                with runtime.state_lock:
                    enabled = runtime.enabled
                if enabled and arm_error_active(state.err):
                    self._fault_arm(runtime, f"arm state error: {state.err}")
            except Exception as exc:
                if self._closing or not rclpy.ok():
                    return
                with runtime.state_lock:
                    enabled = runtime.enabled
                if enabled:
                    self._fault_arm(runtime, f"state query failed: {exc}")
                else:
                    self._log_throttled(
                        f"state_{runtime.name}",
                        f"{runtime.name} state query failed: {exc}",
                        2.0,
                        error=True,
                    )

    def _publish_state(self, arm_name: str, state) -> None:
        stamp = self.get_clock().now().to_msg()
        joint_message = JointState()
        joint_message.header.stamp = stamp
        joint_message.name = [f"{arm_name}_joint_{index}" for index in range(1, len(state.joints) + 1)]
        joint_message.position = [math.radians(value) for value in state.joints]
        self.joint_publishers[arm_name].publish(joint_message)

        pose = state.pose.as_list()
        quaternion = Rotation.from_euler("xyz", pose[3:6]).as_quat()
        legacy_message = JointState()
        legacy_message.header.stamp = stamp
        legacy_message.position = [
            pose[0] * 1000.0,
            pose[1] * 1000.0,
            pose[2] * 1000.0,
            quaternion[3],
            quaternion[0],
            quaternion[1],
            quaternion[2],
        ]
        self.legacy_tcp_publishers[arm_name].publish(legacy_message)

        pose_message = PoseStamped()
        pose_message.header.stamp = stamp
        pose_message.header.frame_id = f"{arm_name}_arm_base"
        pose_message.pose.position.x = pose[0]
        pose_message.pose.position.y = pose[1]
        pose_message.pose.position.z = pose[2]
        pose_message.pose.orientation.x = quaternion[0]
        pose_message.pose.orientation.y = quaternion[1]
        pose_message.pose.orientation.z = quaternion[2]
        pose_message.pose.orientation.w = quaternion[3]
        self.pose_publishers[arm_name].publish(pose_message)

    def _fault_arm(self, runtime: ArmRuntime, reason: str) -> None:
        with runtime.state_lock:
            if not self.dry_run:
                runtime.fault_latched = True
                runtime.fault_reason = reason
        self.get_logger().error(f"{runtime.name} arm disabled: {reason}")
        self._disable_arm(runtime, reason, stop=True)
        if self.trigger_control_enabled:
            self._request_converter(runtime.name, False)

    def _disable_arm(
        self,
        runtime: ArmRuntime,
        reason: str,
        stop: bool,
        slow_stop: bool = False,
    ) -> None:
        with runtime.state_lock:
            was_enabled = runtime.enabled
            runtime.enabled = False
            runtime.enable_pending = False
            runtime.active_mode = None
            runtime.pending_mode = None
            runtime.latest_target = None
            runtime.last_command_at = None
            runtime.initial_target_accepted = False
            runtime.motion_active = False
            runtime.last_send_at = None
            runtime.last_send_duration_seconds = 0.0
            runtime.send_gap_violation_count = 0
            runtime.limiter_linear_velocity = [0.0, 0.0, 0.0]
            runtime.limiter_angular_velocity = [0.0, 0.0, 0.0]
        if stop and was_enabled and not self.dry_run:
            with runtime.send_lock:
                try:
                    if slow_stop:
                        runtime.client.move_slow_stop()
                    else:
                        runtime.client.move_stop()
                except Exception as exc:
                    stop_kind = "slow" if slow_stop else "immediate"
                    self.get_logger().error(
                        f"{runtime.name} {stop_kind} stop failed: {exc}"
                    )
                if not slow_stop:
                    try:
                        runtime.client.clear_current_trajectory()
                    except Exception as exc:
                        self.get_logger().debug(
                            f"{runtime.name} clear trajectory failed: {exc}"
                        )
                    self._sync_runtime_to_actual(runtime)
                else:
                    self._sync_runtime_to_actual(runtime)
        if was_enabled:
            self.get_logger().warn(f"{runtime.name} arm disabled: {reason}")

    def _sync_runtime_to_actual(self, runtime: ArmRuntime) -> None:
        try:
            state = runtime.client.get_state()
            pose = state.pose.as_list()
            with runtime.state_lock:
                runtime.anchor_pose = pose
                runtime.last_sent_pose = pose
                runtime.current_joints_rad = [
                    math.radians(value) for value in state.joints
                ]
                runtime.latest_target = None
                runtime.last_command_at = None
            if runtime.name in getattr(self, "pose_publishers", {}):
                self._publish_state(runtime.name, state)
            self.get_logger().debug(f"{runtime.name} target synced to actual pose {pose}")
        except Exception as exc:
            self.get_logger().error(f"{runtime.name} target sync failed: {exc}")

    def _clear_fault_callback(self, arm_name: str, request, response):
        runtime = self.arms[arm_name]
        if not request.data:
            response.success = False
            response.message = "set data=true to clear the latched fault"
            return response
        try:
            runtime.client.move_stop()
            runtime.client.clear_current_trajectory()
            runtime.client.clear_system_error()
            time.sleep(0.1)
            state = runtime.client.get_state()
            if arm_error_active(state.err):
                raise RuntimeError(f"arm error remains: {state.err}")
            with runtime.state_lock:
                runtime.fault_latched = False
                runtime.fault_reason = ""
            self._sync_runtime_to_actual(runtime)
            response.success = True
            response.message = f"{arm_name} fault cleared"
            self.get_logger().warn(response.message)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _emergency_stop_callback(self, request, response):
        if not request.data:
            response.success = False
            response.message = "set data=true to request emergency stop"
            return response
        errors = []
        for runtime in self.arms.values():
            with runtime.state_lock:
                runtime.enabled = False
                runtime.enable_pending = False
                runtime.active_mode = None
                runtime.pending_mode = None
                runtime.motion_active = False
                runtime.last_send_at = None
                runtime.last_send_duration_seconds = 0.0
                runtime.send_gap_violation_count = 0
                if not self.dry_run:
                    runtime.fault_latched = True
                    runtime.fault_reason = "emergency stop"
            if self.trigger_control_enabled:
                self._request_converter(runtime.name, False)
            if not self.dry_run:
                with runtime.send_lock:
                    try:
                        runtime.client.move_stop()
                    except Exception as exc:
                        errors.append(f"{runtime.name}: {exc}")
        response.success = not errors
        response.message = "emergency stop requested" if not errors else "; ".join(errors)
        self.get_logger().error(response.message)
        return response

    def _log_throttled(self, key: str, message: str, interval: float, error: bool = False) -> None:
        now = time.monotonic()
        if now - self._last_log_times.get(key, 0.0) < interval:
            return
        self._last_log_times[key] = now
        if error:
            self.get_logger().error(message)
        else:
            self.get_logger().info(message)

    @staticmethod
    def _rounded(values: List[float]) -> List[float]:
        return [round(value, 5) for value in values]

    def _disconnect_all(self) -> None:
        for runtime in self.arms.values():
            try:
                runtime.client.disconnect()
            except Exception:
                pass

    def destroy_node(self):
        if not self._closing:
            self._closing = True
            self._sender_stop.set()
            for runtime in self.arms.values():
                self._disable_arm(runtime, "node shutdown", stop=True)
            for thread in self._sender_threads:
                thread.join(timeout=2.0)
            self._disconnect_all()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RealmanDualArmBridge()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
