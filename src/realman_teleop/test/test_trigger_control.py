from std_msgs.msg import String

from realman_teleop.bridge import (
    ArmRuntime,
    RealmanDualArmBridge,
    requested_hold_mode,
)


def make_bridge(mode="hold"):
    bridge = object.__new__(RealmanDualArmBridge)
    runtime = ArmRuntime(name="left", client=object())
    bridge.arms = {"left": runtime}
    bridge.control_allowed = {"left": True}
    bridge.trigger_mode = mode
    bridge.orientation_control_enabled = True
    actions = []

    def request_enable(arm, control_mode):
        arm.enable_pending = True
        arm.pending_mode = control_mode
        actions.append(("enable", control_mode))

    def request_disable(arm, reason):
        arm.enabled = False
        arm.enable_pending = False
        arm.active_mode = None
        arm.pending_mode = None
        actions.append(("disable", reason))

    def request_switch(arm, control_mode):
        arm.enabled = False
        arm.enable_pending = True
        arm.active_mode = None
        arm.pending_mode = control_mode
        actions.append(("switch", control_mode))

    bridge._request_trigger_enable = request_enable
    bridge._request_trigger_disable = request_disable
    bridge._request_trigger_mode_switch = request_switch
    return bridge, runtime, actions


def button(value):
    message = String()
    message.data = value
    return message


def test_requested_hold_mode_covers_single_and_combined_control():
    assert requested_hold_mode(False, False) is None
    assert requested_hold_mode(True, False) == "position"
    assert requested_hold_mode(False, True) == "orientation"
    assert requested_hold_mode(True, True) == "combined"


def test_hold_position_press_and_release():
    bridge, _, actions = make_bridge()
    bridge._mode_button_callback("left", "position", "LT", button("LT=T"))
    bridge._mode_button_callback("left", "position", "LT", button("LT=T"))
    bridge._mode_button_callback("left", "position", "LT", button("LT=F"))
    assert actions == [
        ("enable", "position"),
        ("disable", "position button released"),
    ]


def test_second_button_promotes_pending_enable_to_combined():
    bridge, runtime, actions = make_bridge()
    bridge._mode_button_callback("left", "position", "LT", button("LT=T"))
    bridge._mode_button_callback("left", "orientation", "LG", button("LG=T"))
    assert actions == [("enable", "position")]
    assert runtime.pending_mode == "combined"


def test_completed_pending_enable_uses_latest_combined_mode():
    bridge, runtime, actions = make_bridge()
    runtime.position_pressed = True
    runtime.orientation_pressed = True
    runtime.enable_pending = True
    runtime.pending_mode = "combined"
    bridge._enable_runtime = lambda arm, mode: (
        actions.append(("complete", mode)) or True,
        "ok",
    )
    bridge._request_converter = lambda *_args, **_kwargs: None

    bridge._complete_trigger_enable(runtime, "position", True, "ok")

    assert actions == [("complete", "combined")]


def test_active_single_mode_switches_to_combined():
    bridge, runtime, actions = make_bridge()
    runtime.enabled = True
    runtime.active_mode = "position"
    runtime.position_pressed = True

    bridge._mode_button_callback("left", "orientation", "LG", button("LG=T"))

    assert actions == [("switch", "combined")]


def test_releasing_one_button_switches_combined_to_remaining_mode():
    bridge, runtime, actions = make_bridge()
    runtime.enabled = True
    runtime.active_mode = "combined"
    runtime.position_pressed = True
    runtime.orientation_pressed = True

    bridge._mode_button_callback("left", "orientation", "LG", button("LG=F"))

    assert actions == [("switch", "position")]


def test_releasing_last_button_stops_control():
    bridge, runtime, actions = make_bridge()
    runtime.enabled = True
    runtime.active_mode = "position"
    runtime.position_pressed = True

    bridge._mode_button_callback("left", "position", "LT", button("LT=F"))

    assert actions == [("disable", "position button released")]


def test_mode_switch_slow_stops_and_recaptures_combined_anchor():
    bridge = object.__new__(RealmanDualArmBridge)
    bridge.trigger_mode = "hold"
    bridge.slow_stop_on_trigger_release = True
    runtime = ArmRuntime(name="left", client=object())
    runtime.enabled = True
    runtime.active_mode = "position"
    runtime.position_pressed = True
    runtime.orientation_pressed = True
    actions = []

    def disable(arm, reason, stop, slow_stop=False):
        actions.append(("disable", reason, stop, slow_stop))
        arm.enabled = False
        arm.active_mode = None
        arm.enable_pending = False
        arm.pending_mode = None

    def converter(_arm_name, enabled, callback=None):
        actions.append(("converter", enabled))
        if callback is not None:
            callback(True, "ok")

    bridge._disable_arm = disable
    bridge._reset_interpolator = lambda arm_name: actions.append(
        ("reset", arm_name)
    )
    bridge._request_converter = converter
    bridge._enable_runtime = lambda arm, mode: (
        actions.append(("enable", mode)) or True,
        "ok",
    )

    bridge._request_trigger_mode_switch(runtime, "combined")

    assert actions == [
        ("disable", "control mode position->combined", True, True),
        ("reset", "left"),
        ("converter", True),
        ("enable", "combined"),
    ]


def test_toggle_mode_keeps_conflict_stop_behavior():
    bridge, runtime, actions = make_bridge("toggle")
    runtime.enabled = True
    runtime.active_mode = "position"
    runtime.position_pressed = True

    bridge._mode_button_callback("left", "orientation", "LG", button("LG=T"))

    assert actions == [("disable", "position/orientation conflict")]
