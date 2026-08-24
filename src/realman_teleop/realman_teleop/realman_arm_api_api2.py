
from __future__ import annotations

"""
睿尔曼机械臂高层 API 封装（API2 / Python 版）


底层 SDK 导入路径：
    Robotic_Arm.rm_robot_interface

当前封装的主要能力：
- 机械臂连接 / 断开连接
- 电源控制、运行模式、状态查询
- 轨迹控制：movej / movej_p / movel / movec
- 实时透传（CANFD）：movej_canfd / movep_canfd
- 运动暂停 / 继续 / 停止 / 清轨迹
- 末端力传感器 / 拖动示教
- 睿尔曼原生夹爪
- 工具端电源 / RS485 Modbus
- 大寰夹爪（DH）
- 钧舵夹爪（JD）
- 升降柱
- UDP 主动上报
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass
import importlib
import logging
import math
import time
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


# 模块级日志对象。
# 如果外部配置了 logging，这个 logger 会自动继承配置。
LOGGER = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 类型别名
# ----------------------------------------------------------------------
# PoseLike:
#   表示“像位姿一样的数据”。
#   本质上是一个长度为 6 的序列，通常格式为：
#       [x, y, z, rx, ry, rz]
#   CANFD 位姿透传也支持长度为 7 的四元数格式：
#       [x, y, z, qw, qx, qy, qz]
PoseLike = Sequence[Union[int, float]]

# JointLike:
#   表示“像关节角一样的数据”。
#   本质上是一个长度为 6 或 7 的序列，通常格式为：
#       [j1, j2, j3, j4, j5, j6]
#   或 7 轴机械臂的 7 个关节值。
JointLike = Sequence[Union[int, float]]


# ----------------------------------------------------------------------
# 数据结构：位姿 / 状态对象
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ArmPose:
    """
    机械臂位姿。

    字段含义：
    - x, y, z : 末端位置，单位通常为米
    - rx, ry, rz : 末端姿态角，单位通常为弧度

    为什么要单独做成 dataclass？
    - 比直接用 list 更清晰
    - 不容易把 x/y/z 顺序写错
    - 便于 IDE 补全和类型提示
    """

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    def as_list(self) -> List[float]:
        """
        把位姿对象转成底层 SDK 常用的 list 形式。

        例如：
            ArmPose(0.3, 0.0, 0.2, 3.14, 0.0, 0.0).as_list()

        会变成：
            [0.3, 0.0, 0.2, 3.14, 0.0, 0.0]
        """
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    @classmethod
    def from_any(cls, pose: Union["ArmPose", PoseLike]) -> "ArmPose":
        """
        把“ArmPose 对象”或“普通序列”统一转成 ArmPose。

        这样上层调用时可以两种写法都支持：
            ArmPose(...)
        或
            [x, y, z, rx, ry, rz]
        """
        if isinstance(pose, ArmPose):
            return pose

        values = [float(v) for v in pose]
        if len(values) != 6:
            raise ValueError("pose length must be 6")
        return cls(*values)


@dataclass(frozen=True)
class ArmState:
    """
    机械臂当前综合状态。

    常见字段：
    - joints : 当前关节角列表
    - pose   : 当前末端位姿
    - err    : 当前错误字典（保留原始字段，便于排查问题）
    """

    joints: List[float]
    pose: ArmPose
    err: Dict[str, Any]


@dataclass(frozen=True)
class ControllerState:
    """
    控制器状态。

    字段说明：
    - voltage     : 控制器电压
    - current     : 控制器电流
    - temperature : 控制器温度
    - sys_err     : 系统错误码
    - raw         : 原始返回字典，调试时可以直接查看底层返回内容
    """

    voltage: float
    current: float
    temperature: float
    sys_err: int
    raw: Dict[str, Any]


@dataclass(frozen=True)
class GripperState:
    """
    睿尔曼原生夹爪状态（不是大寰，不是钧舵）。

    字段说明：
    - enable_state : 是否使能
    - status       : 当前状态
    - error        : 错误码
    - mode         : 模式
    - current_force: 当前力值
    - temperature  : 温度
    - actpos       : 当前实际位置
    - raw          : 原始返回字典
    """

    enable_state: int
    status: int
    error: int
    mode: int
    current_force: int
    temperature: int
    actpos: int
    raw: Dict[str, Any]


@dataclass(frozen=True)
class ForceSensorData:
    """
    六维力传感器数据。

    字段顺序均为：
    [Fx, Fy, Fz, Mx, My, Mz]

    - force_data           : 原始力/力矩，单位 N / Nm
    - zero_force_data      : 传感器坐标系下清零后的外力
    - work_zero_force_data : 工作坐标系下清零后的外力
    - tool_zero_force_data : 工具坐标系下清零后的外力
    """

    force_data: List[float]
    zero_force_data: List[float]
    work_zero_force_data: List[float]
    tool_zero_force_data: List[float]
    raw: Dict[str, Any]


@dataclass(frozen=True)
class FzSensorData:
    """
    一维力传感器数据。

    - fz           : 原始 Z 向力，单位 N
    - zero_fz      : 传感器坐标系下清零后的 Z 向外力
    - work_zero_fz : 工作坐标系下清零后的 Z 向外力
    - tool_zero_fz : 工具坐标系下清零后的 Z 向外力
    """

    fz: float
    zero_fz: float
    work_zero_fz: float
    tool_zero_fz: float
    raw: Dict[str, Any]


@dataclass(frozen=True)
class DHGripperState:
    """
    大寰夹爪状态。

    字段说明：
    - init_status      : 初始化状态
        0 = 未初始化
        1 = 初始化完成
    - grip_status      : 夹持状态
        0 = 运动中
        1 = 到位未夹物
        2 = 夹住物体
        3 = 物体掉落
    - current_position : 当前实际位置（通常 0~1000）
    """

    init_status: int
    grip_status: int
    current_position: int

    @property
    def init_status_text(self) -> str:
        """把初始化状态码翻译成人类可读的中文。"""
        return {
            0: "未初始化",
            1: "初始化完成",
        }.get(self.init_status, "未知状态(%s)" % self.init_status)

    @property
    def grip_status_text(self) -> str:
        """把夹持状态码翻译成人类可读的中文。"""
        return {
            0: "运动中",
            1: "到位未夹物",
            2: "夹住物体",
            3: "物体掉落",
        }.get(self.grip_status, "未知状态(%s)" % self.grip_status)


@dataclass(frozen=True)
class JDGripperState:
    """
    钧舵夹爪状态。

    这些字段来自钧舵状态寄存器解析结果。
    这里尽量保留手册里的位域语义，方便你后续继续对照协议学习。

    字段说明：
    - gact       : 激活位
    - gdrop_sta  : 掉落状态
    - gmode      : 模式位
    - ggto       : 目标动作位（是否正在向目标位置运动）
    - gsta       : 夹爪总状态
    - gobj       : 夹持检测状态
    - fault_code : 故障码
    - current_position : 当前位置
    - current_speed    : 当前速度
    - current_force    : 当前力
    - bus_voltage      : 总线电压
    - temperature      : 温度
    """

    gact: int
    gdrop_sta: int
    gmode: int
    ggto: int
    gsta: int
    gobj: int
    fault_code: int
    current_position: int
    current_speed: int
    current_force: int
    bus_voltage: int
    temperature: int

    @property
    def active(self) -> bool:
        """是否处于已使能状态。"""
        return self.gact == 1

    @property
    def activated(self) -> bool:
        """是否已经完成激活。一般 gsta == 3 表示激活完成。"""
        return self.gsta == 3

    @property
    def holding(self) -> bool:
        """是否夹住了物体。一般 gobj == 2 表示闭合夹到物体。"""
        return self.gobj == 2

    @property
    def dropped(self) -> bool:
        """是否发生了物体掉落。"""
        return self.gdrop_sta == 1 or self.gobj == 3

    @property
    def gsta_text(self) -> str:
        """gsta 的中文解释。"""
        return {
            0: "复位/巡检中",
            1: "激活中",
            3: "激活完成",
        }.get(self.gsta, "未知状态(%s)" % self.gsta)

    @property
    def gobj_text(self) -> str:
        """gobj 的中文解释。"""
        return {
            0: "运动中",
            1: "张开时遇物停止",
            2: "闭合时夹到物体",
            3: "到位未检测到物体或物体掉落",
        }.get(self.gobj, "未知状态(%s)" % self.gobj)


@dataclass(frozen=True)
class LiftState:
    """
    升降柱状态。

    字段说明：
    - height : 当前高度
    - current: 当前电流
    - err    : 错误码
    - mode   : 当前模式
    - raw    : 原始返回数据
    """

    height: int
    current: int
    err: int
    mode: int
    raw: Dict[str, Any]

    @property
    def mode_text(self) -> str:
        """把 mode 数字翻译成中文状态。"""
        return {
            0: "空闲",
            1: "正向速度运动",
            2: "正向位置运动",
            3: "负向速度运动",
            4: "负向位置运动",
        }.get(self.mode, "未知模式(%s)" % self.mode)


# ----------------------------------------------------------------------
# 自定义异常
# ----------------------------------------------------------------------
class ArmApiError(RuntimeError):
    """
    当底层 API2 返回非 0 状态码时抛出的统一异常。

    设计原因：
    - 不希望上层每次都手动写 if ret != 0
    - 所以本封装统一把“错误状态码”转成 Python 异常
    - 上层只需要 try / except 就可以处理错误
    """

    def __init__(
        self,
        code: int,
        message: str,
        *,
        api_name: Optional[str] = None,
        vendor_result: Any = None,
    ) -> None:
        self.code = int(code)
        self.message = message
        self.api_name = api_name
        self.vendor_result = vendor_result

        prefix = "%s: " % api_name if api_name else ""
        super().__init__("%s[%s] %s" % (prefix, self.code, self.message))


# ----------------------------------------------------------------------
# 主封装类
# ----------------------------------------------------------------------
class RealmanArmClient(AbstractContextManager):
    """
    睿尔曼机械臂高层客户端（基于 API2 / Python）。

    这个类是项目中的主要入口对象，负责：
    - 建立连接 / 断开连接
    - 机械臂运动控制
    - 状态查询
    - 原生夹爪控制
    - 外部夹爪（大寰 / 钧舵）控制
    - 升降柱控制
    - Modbus RTU 通信
    """

    # 兼容包模式和旧版脚本模式下的 API2 Python SDK 路径。
    SDK_IMPORT_PATHS = (
        "arm_api_new.Robotic_Arm.rm_robot_interface",
        "Robotic_Arm.rm_robot_interface",
    )

    # -----------------------------
    # 大寰夹爪常用寄存器
    # -----------------------------
    _DH_REG_INIT_CMD = 0x0100
    _DH_REG_FORCE = 0x0101
    _DH_REG_TARGET_POSITION = 0x0103
    _DH_REG_SPEED = 0x0104
    _DH_REG_INIT_STATE = 0x0200
    _DH_REG_GRIP_STATE = 0x0201
    _DH_REG_CURRENT_POSITION = 0x0202
    _DH_REG_IO_MODE_SWITCH = 0x0402

    # -----------------------------
    # 钧舵夹爪常用寄存器
    # -----------------------------
    _JD_REG_CTRL = 0x03E8
    _JD_REG_POS = 0x03E9
    _JD_REG_SPEED_FORCE = 0x03EA
    _JD_REG_STATUS = 0x07D0
    _JD_REG_FAULT_POS = 0x07D1
    _JD_REG_VOLT_TEMP = 0x07D3
    _JD_DEFAULT_DEVICE = 9

    def __init__(
        self,
        ip: str = "192.168.1.18",
        *,
        port: int = 8080,
        model: str = "RM65",
        dof: Optional[int] = None,
        auto_connect: bool = False,
        logger: Optional[logging.Logger] = None,
        thread_mode: str = "triple",
        log_level: int = 3,
    ) -> None:
        """
        初始化客户端对象。

        参数说明：
        - ip           : 机械臂控制器 IP
        - port         : 机械臂控制器端口，默认 8080
        - model        : 型号字符串，仅用于项目侧标记
        - dof          : 自由度数量（如 6 / 7）
        - auto_connect : 是否在创建对象后立即 connect()
        - logger       : 外部自定义日志对象
        - thread_mode  : API2 线程模式
        - log_level    : SDK 日志等级
        """
        self.ip = ip
        self.port = int(port)
        self.model = model
        self.dof = dof
        self.logger = logger or LOGGER
        self.thread_mode = thread_mode
        self.log_level = int(log_level)

        # 延迟导入后的 SDK 模块对象
        self._vendor_module = None

        # API2 RoboticArm 对象
        self._arm = None

        # 连接成功后返回的机械臂句柄
        self._handle = None

        # 用于串行化 SDK 调用，避免多线程同时访问同一个连接对象
        self._lock = Lock()

        if auto_connect:
            self.connect()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def __enter__(self) -> "RealmanArmClient":
        """支持 with 语法。"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """退出 with 块时自动断开连接。"""
        self.disconnect()

    @property
    def connected(self) -> bool:
        """当前是否已经成功建立连接。"""
        return self._arm is not None and self._handle is not None

    def connect(self) -> "RealmanArmClient":
        """
        建立机械臂连接。

        内部流程：
        1. 延迟导入 SDK
        2. 解析线程模式
        3. 创建 RoboticArm 对象
        4. 调用 rm_create_robot_arm(ip, port, log_level)
        5. 检查句柄 id，若失败则抛异常
        """
        if self.connected:
            return self

        vendor = self._load_vendor_module()
        mode_value = self._resolve_thread_mode(vendor, self.thread_mode)

        # 创建 API2 机械臂对象
        self._arm = vendor.RoboticArm(mode_value)

        # 真正发起连接
        with self._lock:
            handle = self._arm.rm_create_robot_arm(self.ip, self.port, self.log_level)

        # API2 返回的 handle 一般带有 id 字段，id < 0 通常表示连接失败
        handle_id = getattr(handle, "id", None)
        if handle_id is None or int(handle_id) < 0:
            raise ArmApiError(
                int(handle_id if handle_id is not None else -1),
                "创建机械臂连接失败",
                api_name="rm_create_robot_arm",
                vendor_result=handle,
            )

        self._handle = handle
        return self

    def disconnect(self) -> None:
        """
        断开机械臂连接。

        这里采用“尽力而为”的处理方式：
        - 即使断开时报错，也尽量不影响主程序退出流程
        """
        if self._arm is None:
            return

        try:
            with self._lock:
                self._arm.rm_delete_robot_arm()
        except Exception:
            self.logger.debug("rm_delete_robot_arm failed during disconnect", exc_info=True)

        self._handle = None
        self._arm = None

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _load_vendor_module(self):
        """
        延迟导入 SDK 模块。

        为什么要这样做？
        - 如果机器上还没有正确放 SDK
        - import 本文件时不至于立刻崩溃
        - 只有真正 connect() 时才要求 SDK 环境完整
        """
        if self._vendor_module is None:
            last_error = None
            for module_path in self.SDK_IMPORT_PATHS:
                try:
                    self._vendor_module = importlib.import_module(module_path)
                    break
                except ImportError as exc:
                    last_error = exc
            if self._vendor_module is None:
                raise ImportError(
                    "无法导入 Robotic_Arm.rm_robot_interface，请先将新版 API2 Python 开发包放在 "
                    "arm_api_new/Robotic_Arm。"
                ) from last_error
        return self._vendor_module

    @staticmethod
    def _resolve_thread_mode(vendor: Any, thread_mode: Union[str, Any]) -> Any:
        """
        把字符串形式的线程模式转成 API2 枚举值。

        支持：
        - "single"
        - "dual"
        - "triple"

        如果传进来的本身已经是底层枚举值，则直接返回。
        """
        if not isinstance(thread_mode, str):
            return thread_mode

        value = thread_mode.strip().lower()
        enum_cls = vendor.rm_thread_mode_e

        if value in ("single", "rm_single_mode_e"):
            return enum_cls.RM_SINGLE_MODE_E
        if value in ("dual", "rm_dual_mode_e"):
            return enum_cls.RM_DUAL_MODE_E

        # 默认使用三线程模式
        return enum_cls.RM_TRIPLE_MODE_E

    def _require_arm(self):
        """
        确保当前已经连接。

        这是很多内部函数的前置检查：
        如果还没 connect()，直接拒绝继续执行。
        """
        if not self.connected:
            raise RuntimeError("robot arm is not connected")
        return self._arm

    @staticmethod
    def _extract_status_from_dict(data: Dict[str, Any]) -> Optional[int]:
        """
        尝试从 dict 返回值中提取“状态码”。

        不同 API2 接口返回的字典字段命名不完全一致，
        所以这里统一尝试常见字段名：
        - return_code
        - ret
        - code
        - err_code
        - status
        """
        for key in ("return_code", "ret", "code", "err_code", "status"):
            if key in data and isinstance(data[key], int):
                return int(data[key])
        return None

    def _normalize_result(self, api_name: str, result: Any) -> Any:
        """
        统一规范底层 API2 返回结果。

        目标：
        - 控制类接口：成功时返回 None
        - 查询类接口：成功时返回 payload
        - 失败时统一抛 ArmApiError

        支持的常见返回形式：
        1. int
           - 0 表示成功
           - 非 0 表示失败

        2. tuple
           - 常见形式为 (ret, payload)
           - 或 (ret, v1, v2, ...)

        3. dict
           - 某些接口直接返回字典
        """
        if isinstance(result, int):
            if result != 0:
                raise ArmApiError(result, "API2 调用失败", api_name=api_name, vendor_result=result)
            return None

        if isinstance(result, tuple):
            if not result:
                return tuple()

            ret = result[0]
            if isinstance(ret, int) and ret != 0:
                raise ArmApiError(ret, "API2 调用失败", api_name=api_name, vendor_result=result)

            if len(result) == 1:
                return tuple()
            if len(result) == 2:
                return result[1]
            return result[1:]

        if isinstance(result, dict):
            ret = self._extract_status_from_dict(result)
            if ret is not None and ret != 0:
                raise ArmApiError(ret, "API2 调用失败", api_name=api_name, vendor_result=result)
            return result

        return result

    def raw_call(self, method_name: str, *args: Any, check: bool = True) -> Any:
        """
        直接调用底层任意 API2 方法。

        这是一个“逃生口”：
        - 如果你发现某个新接口还没封装
        - 可以先用 raw_call("rm_xxx", ...) 直接访问
        """
        arm = self._require_arm()

        if not hasattr(arm, method_name):
            raise AttributeError("API2 SDK missing method: %s" % method_name)

        method = getattr(arm, method_name)

        # 所有底层调用都加锁，避免多线程并发访问同一连接对象
        with self._lock:
            result = method(*args)

        return self._normalize_result(method_name, result) if check else result

    def _call_void(self, method_name: str, *args: Any) -> None:
        """调用控制类接口。"""
        self.raw_call(method_name, *args, check=True)

    def _call_get(self, method_name: str, *args: Any) -> Any:
        """调用查询类接口。"""
        return self.raw_call(method_name, *args, check=True)

    @staticmethod
    def _validate_joint_target(joints: JointLike, dof: Optional[int] = None) -> List[float]:
        """
        校验关节目标数组。

        作用：
        1. 把输入统一转成 float 列表
        2. 检查长度是否是 6 或 7
        3. 如果用户显式指定了 dof，则进一步检查长度是否一致
        """
        values = [float(v) for v in joints]

        if dof is not None and len(values) != int(dof):
            raise ValueError("expected %s joint values, got %s" % (dof, len(values)))

        if len(values) not in (6, 7):
            raise ValueError("joint target length must be 6 or 7")

        return values

    @staticmethod
    def _validate_canfd_pose_target(pose: Union[ArmPose, PoseLike]) -> List[float]:
        """
        校验 CANFD 位姿透传目标。

        普通运动接口只使用 6 元欧拉角位姿；底层 movep_canfd 还支持
        7 元四元数位姿 [x, y, z, qw, qx, qy, qz]。
        """
        if isinstance(pose, ArmPose):
            return pose.as_list()

        values = [float(v) for v in pose]
        if len(values) not in (6, 7):
            raise ValueError("CANFD pose target length must be 6(euler) or 7(quaternion)")
        return values

    @staticmethod
    def _validate_canfd_options(trajectory_mode: int, radio: int) -> Tuple[int, int]:
        """校验 CANFD 高跟随模式下的平滑参数。"""
        mode = int(trajectory_mode)
        if mode not in (0, 1, 2):
            raise ValueError("trajectory_mode must be 0(passthrough), 1(curve), or 2(filter)")

        radio_value = int(radio)
        radio_max = 999 if mode == 1 else 100
        if not (0 <= radio_value <= radio_max):
            raise ValueError("radio must be in [0, %s] for trajectory_mode=%s" % (radio_max, mode))
        return mode, radio_value

    @staticmethod
    def _validate_percent(name: str, value: Union[int, float], minimum: float = 0.0, maximum: float = 100.0) -> int:
        """
        校验百分比参数，并返回 int。

        为什么统一转 int？
        - 因为底层不少接口更喜欢收到 int
        - 你前面已经遇到过 float 导致的参数类型报错
        """
        numeric = float(value)
        if not (minimum <= numeric <= maximum):
            raise ValueError("%s must be in [%s, %s], got %s" % (name, minimum, maximum, value))
        return int(numeric)

    @staticmethod
    def _validate_range(name: str, value: Union[int, float], minimum: float, maximum: float) -> int:
        """通用范围校验。"""
        numeric = float(value)
        if not (minimum <= numeric <= maximum):
            raise ValueError("%s must be in [%s, %s], got %s" % (name, minimum, maximum, value))
        return int(numeric)

    def _make_modbus_param(self, port: int, address: int, device: int, num: Optional[int] = None) -> Any:
        """
        构造 Modbus 参数结构体 rm_peripheral_read_write_params_t。

        API2 的 Modbus 接口通常不是直接传 port/address/device，
        而是先构造一个结构体对象，再把结构体传给底层函数。
        """
        vendor = self._load_vendor_module()
        cls = vendor.rm_peripheral_read_write_params_t

        if num is None:
            return cls(int(port), int(address), int(device))

        return cls(int(port), int(address), int(device), int(num))

    @staticmethod
    def _to_signed_8bit(value: int) -> int:
        """
        无符号 8 位整数转有符号 8 位整数。

        常用于解析协议里“温度字节”这类字段。
        """
        value &= 0xFF
        return value - 256 if value >= 128 else value

    @staticmethod
    def _as_float_list(value: Any, *, length: Optional[int] = None) -> List[float]:
        """把底层 ctypes 数组 / 普通序列统一转成 float list。"""
        values = [float(v) for v in value]
        if length is not None and len(values) != int(length):
            raise RuntimeError("unexpected float list length: %r" % (values,))
        return values

    # ------------------------------------------------------------------
    # 外部夹爪公共初始化 / 反初始化
    # ------------------------------------------------------------------
    def init_external_gripper(
        self,
        *,
        port: int = 1,
        baudrate: int = 115200,
        timeout: int = 3,
        tool_voltage_type: int = 3,
        switch_to_modbus: bool = True,
    ) -> None:
        """
        初始化“非默认夹爪”（例如大寰、钧舵等外接夹爪）的公共环境。

        这个函数不关心你接的是哪种夹爪，它做的是“公共准备工作”：
        1. 给工具端上电
        2. 把对应 RS485 口切到 Modbus RTU 模式

        参数说明：
        - port:
            使用哪个 RS485 口，常见外接末端夹爪一般是 1
        - baudrate:
            通讯波特率，外接夹爪常见是 115200
        - timeout:
            Modbus 通讯超时
        - tool_voltage_type:
            工具端电压，常见 3 = 24V
        - switch_to_modbus:
            是否执行 RS485 -> Modbus 模式切换

        使用建议：
        - 在控制外部夹爪前先调用一次这个函数
        """
        # 1) 工具端上电
        self.set_tool_voltage(tool_voltage_type)

        # 2) 切换 RS485 到 Modbus RTU 主站模式
        if switch_to_modbus:
            self.set_modbus_mode(port=port, baudrate=baudrate, timeout=timeout)

    def deinit_external_gripper(
        self,
        *,
        power_off: bool = True,
        tool_voltage_type: int = 0,
    ) -> None:
        """
        反初始化外部夹爪公共环境。

        这里不去做“恢复成其他通信模式”这类复杂动作，
        只做最常见、最有用的操作：关闭工具端供电。

        参数说明：
        - power_off:
            是否断开工具端供电
        - tool_voltage_type:
            一般为 0，表示关闭输出
        """
        if power_off:
            self.set_tool_voltage(tool_voltage_type)

    # ------------------------------------------------------------------
    # 基础信息 / 电源 / 模式
    # ------------------------------------------------------------------
    def api_version(self) -> str:
        """
        获取一个版本说明字符串。

        API2 没有和旧版完全同名的 API_Version()，
        这里优先尝试读软件信息，如果失败就返回一个固定说明。
        """
        try:
            info = self.get_arm_software_info()
            return str(info)
        except Exception:
            return "API2(Python)"

    def socket_state(self) -> int:
        """兼容旧接口风格：1=已连接，0=未连接。"""
        return 1 if self.connected else 0

    def get_robot_info(self) -> Dict[str, Any]:
        """获取机器人信息。"""
        return self._call_get("rm_get_robot_info")

    def get_arm_software_info(self) -> Dict[str, Any]:
        """获取机械臂软件信息。"""
        return self._call_get("rm_get_arm_software_info")

    def set_power(self, enabled: bool) -> None:
        """设置机械臂上电 / 下电。"""
        self._call_void("rm_set_arm_power", 1 if enabled else 0)

    def power_on(self, *, block: bool = True) -> None:
        """机械臂上电。"""
        self.set_power(True)

    def power_off(self, *, block: bool = True) -> None:
        """机械臂下电。"""
        self.set_power(False)

    def get_power_state(self) -> bool:
        """获取机械臂当前是否上电。"""
        return bool(int(self._call_get("rm_get_arm_power_state")))

    def get_run_mode(self) -> int:
        """获取机械臂运行模式。"""
        return int(self._call_get("rm_get_arm_run_mode"))

    def set_run_mode(self, mode: int) -> None:
        """设置机械臂运行模式。"""
        if mode not in (0, 1):
            raise ValueError("mode must be 0(simulation) or 1(real)")
        self._call_void("rm_set_arm_run_mode", int(mode))

    def get_controller_state(self) -> ControllerState:
        """获取控制器状态。"""
        data = self._call_get("rm_get_controller_state")
        if not isinstance(data, dict):
            raise RuntimeError("unexpected rm_get_controller_state payload: %r" % (data,))
        return ControllerState(
            voltage=float(data.get("voltage", 0.0)),
            current=float(data.get("current", 0.0)),
            temperature=float(data.get("temperature", 0.0)),
            sys_err=int(data.get("sys_err", 0)),
            raw=data,
        )

    # ------------------------------------------------------------------
    # 机械臂状态
    # ------------------------------------------------------------------
    def get_state(self) -> ArmState:
        """获取机械臂当前综合状态。"""
        data = self._call_get("rm_get_current_arm_state")
        if not isinstance(data, dict):
            raise RuntimeError("unexpected rm_get_current_arm_state payload: %r" % (data,))
        pose_raw = data.get("pose", [0, 0, 0, 0, 0, 0])
        joint_raw = data.get("joint", [])
        err_raw = data.get("err", {})
        return ArmState(
            joints=[float(v) for v in joint_raw],
            pose=ArmPose.from_any(pose_raw),
            err=dict(err_raw) if isinstance(err_raw, dict) else {"raw": err_raw},
        )

    def clear_system_error(self) -> None:
        """清除系统错误/报警状态（对应碰撞保护、超限等触发的故障）。"""
        self._call_void("rm_clear_system_err")

    def get_joint_temperatures(self) -> List[float]:
        """获取各关节温度列表。"""
        return [float(v) for v in self._call_get("rm_get_current_joint_temperature")]

    def get_joint_currents(self) -> List[float]:
        """获取各关节电流列表。"""
        return [float(v) for v in self._call_get("rm_get_current_joint_current")]

    def get_joint_voltages(self) -> List[float]:
        """获取各关节电压列表。"""
        return [float(v) for v in self._call_get("rm_get_current_joint_voltage")]

    def get_joint_degrees(self) -> List[float]:
        """获取各关节角度列表。"""
        return [float(v) for v in self._call_get("rm_get_joint_degree")]

    def get_joint_position_limits(self) -> Tuple[List[float], List[float]]:
        """获取各关节软限位，返回 ``(min_deg, max_deg)``。"""
        min_positions = [float(v) for v in self._call_get("rm_get_joint_min_pos")]
        max_positions = [float(v) for v in self._call_get("rm_get_joint_max_pos")]
        expected_dof = self.dof or min(len(min_positions), len(max_positions))
        return min_positions[:expected_dof], max_positions[:expected_dof]

    def get_current_work_frame(self) -> Dict[str, Any]:
        """获取控制器当前激活的工作坐标系。"""
        data = self._call_get("rm_get_current_work_frame")
        if not isinstance(data, dict):
            raise RuntimeError("unexpected rm_get_current_work_frame payload: %r" % (data,))
        return data

    def get_current_tool_frame(self) -> Dict[str, Any]:
        """获取控制器当前激活的工具坐标系及负载参数。"""
        data = self._call_get("rm_get_current_tool_frame")
        if not isinstance(data, dict):
            raise RuntimeError("unexpected rm_get_current_tool_frame payload: %r" % (data,))
        return data

    def solve_inverse_kinematics(
        self,
        pose: Union[ArmPose, PoseLike],
        *,
        reference_joints: Optional[JointLike] = None,
    ) -> List[float]:
        """以当前或指定关节角为参考，对六元欧拉角位姿进行逆运动学求解。"""
        target = ArmPose.from_any(pose).as_list()
        if reference_joints is None:
            reference = self.get_state().joints
        else:
            reference = self._validate_joint_target(reference_joints, self.dof)

        q_in = [float(value) for value in reference]
        if len(q_in) == 6:
            q_in.append(0.0)

        vendor = self._load_vendor_module()
        params = vendor.rm_inverse_kinematics_params_t(q_in=q_in, q_pose=target, flag=1)
        result = self._call_get("rm_algo_inverse_kinematics", params)
        if not isinstance(result, (list, tuple)):
            raise RuntimeError("unexpected inverse kinematics result: %r" % (result,))
        solution = [float(value) for value in result]
        expected_dof = self.dof or len(reference)
        return solution[:expected_dof]

    def get_all_state(self) -> Dict[str, Any]:
        """获取机械臂完整状态字典。"""
        data = self._call_get("rm_get_arm_all_state")
        if not isinstance(data, dict):
            raise RuntimeError("unexpected rm_get_arm_all_state payload: %r" % (data,))
        return data

    # ------------------------------------------------------------------
    # 轨迹控制
    # ------------------------------------------------------------------
    def movej(
        self,
        joints: JointLike,
        *,
        v: Union[int, float] = 20,
        r: Union[int, float] = 0,
        trajectory_connect: int = 0,
        block: bool = True,
    ) -> None:
        """
        关节空间运动。

        参数：
        - joints             : 目标关节角数组
        - v                  : 速度百分比
        - r                  : 交融半径百分比
        - trajectory_connect : 是否与下一条轨迹连接规划
        - block              : 是否阻塞等待执行完成
        """
        target = self._validate_joint_target(joints, self.dof)
        speed = self._validate_percent("v", v, 1.0, 100.0)
        radius = self._validate_percent("r", r, 0.0, 100.0)
        self._call_void("rm_movej", target, speed, radius, 1 if trajectory_connect else 0, 1 if block else 0)

    def movej_p(
        self,
        pose: Union[ArmPose, PoseLike],
        *,
        v: Union[int, float] = 20,
        r: Union[int, float] = 0,
        trajectory_connect: int = 0,
        block: bool = True,
    ) -> None:
        """
        按目标位姿执行关节空间规划。
        """
        target = ArmPose.from_any(pose).as_list()
        speed = self._validate_percent("v", v, 1.0, 100.0)
        radius = self._validate_percent("r", r, 0.0, 100.0)
        self._call_void("rm_movej_p", target, speed, radius, 1 if trajectory_connect else 0, 1 if block else 0)

    def movel(
        self,
        pose: Union[ArmPose, PoseLike],
        *,
        v: Union[int, float] = 20,
        trajectory_connect: int = 0,
        r: Union[int, float] = 0,
        block: bool = True,
    ) -> None:
        """
        末端直线运动。
        """
        target = ArmPose.from_any(pose).as_list()
        speed = self._validate_percent("v", v, 1.0, 100.0)
        radius = self._validate_percent("r", r, 0.0, 100.0)
        self._call_void("rm_movel", target, speed, radius, 1 if trajectory_connect else 0, 1 if block else 0)

    def movec(
        self,
        pose_via: Union[ArmPose, PoseLike],
        pose_to: Union[ArmPose, PoseLike],
        *,
        v: Union[int, float] = 20,
        loop: int = 0,
        trajectory_connect: int = 0,
        r: Union[int, float] = 0,
        block: bool = True,
    ) -> None:
        """
        圆弧运动。
        """
        via = ArmPose.from_any(pose_via).as_list()
        to = ArmPose.from_any(pose_to).as_list()
        speed = self._validate_percent("v", v, 1.0, 100.0)
        radius = self._validate_percent("r", r, 0.0, 100.0)
        if int(loop) < 0:
            raise ValueError("loop must be >= 0")
        self._call_void("rm_movec", via, to, speed, radius, int(loop), 1 if trajectory_connect else 0, 1 if block else 0)

    def move_home(self, *, block: bool = True) -> None:
        """移动到全零关节角位置。"""
        dof = self.dof if self.dof is not None else 6
        self.movej([0] * dof, block=block)

    # ------------------------------------------------------------------
    # 实时透传（CANFD）
    # ------------------------------------------------------------------
    def movep_canfd(
        self,
        pose: Union[ArmPose, PoseLike],
        *,
        follow: bool = False,
        trajectory_mode: int = 0,
        radio: int = 0,
    ) -> None:
        """
        位姿实时透传（CANFD）。

        跳过规划，直接把目标位姿发给机械臂，用于需要周期性刷新位姿的闭环
        场景（导纳/阻抗控制、视觉伺服等）。

        - follow=True  : 高跟随，要求发送周期 <=10ms，否则运动会不稳定。
        - follow=False : 低跟随，对通信抖动更宽容；调用方无法保证稳定的
          高频率调用时应使用这个。
        - pose 长度为 6 时使用欧拉角 [x,y,z,rx,ry,rz]；长度为 7 时使用
          四元数 [x,y,z,qw,qx,qy,qz]。
        """
        target = self._validate_canfd_pose_target(pose)
        mode, radio_value = self._validate_canfd_options(trajectory_mode, radio)
        self._call_void("rm_movep_canfd", target, bool(follow), mode, radio_value)

    def movej_canfd(
        self,
        joints: JointLike,
        *,
        follow: bool = False,
        expand: float = 0.0,
        trajectory_mode: int = 0,
        radio: int = 0,
    ) -> None:
        """关节角实时透传（CANFD），跟随/抖动要求同 movep_canfd。"""
        target = self._validate_joint_target(joints, self.dof)
        if len(target) == 6:
            target = target + [0.0]
        mode, radio_value = self._validate_canfd_options(trajectory_mode, radio)
        self._call_void("rm_movej_canfd", target, bool(follow), float(expand), mode, radio_value)

    # ------------------------------------------------------------------
    # 运动状态控制
    # ------------------------------------------------------------------
    def move_stop(self) -> None:
        """立即停止机械臂。"""
        self._call_void("rm_set_arm_stop")

    def move_slow_stop(self) -> None:
        """缓慢停止机械臂。"""
        self._call_void("rm_set_arm_slow_stop")

    def move_pause(self) -> None:
        """暂停当前轨迹执行。"""
        self._call_void("rm_set_arm_pause")

    def move_continue(self) -> None:
        """继续已暂停的轨迹。"""
        self._call_void("rm_set_arm_continue")

    def clear_current_trajectory(self) -> None:
        """清除当前轨迹。"""
        self._call_void("rm_set_delete_current_trajectory")

    def clear_all_trajectory(self) -> None:
        """清除全部轨迹。"""
        self._call_void("rm_set_arm_delete_trajectory")

    # ------------------------------------------------------------------
    # 原生夹爪
    # ------------------------------------------------------------------
    def configure_gripper_range(self, min_limit: int = 0, max_limit: int = 1000) -> None:
        """设置原生夹爪有效行程范围。"""
        self._call_void("rm_set_gripper_route", self._validate_range("min_limit", min_limit, 0, 1000), self._validate_range("max_limit", max_limit, 0, 1000))

    def gripper_release(self, speed: int = 500, *, block: bool = True, timeout: int = 5) -> None:
        """原生夹爪张开。"""
        self._call_void("rm_set_gripper_release", self._validate_range("speed", speed, 1, 1000), bool(block), int(timeout))

    def gripper_pick(self, speed: int = 500, force: int = 200, *, block: bool = True, timeout: int = 5) -> None:
        """原生夹爪夹取。"""
        self._call_void("rm_set_gripper_pick", self._validate_range("speed", speed, 1, 1000), self._validate_range("force", force, 50, 1000), bool(block), int(timeout))

    def gripper_pick_keep(self, speed: int = 500, force: int = 200, *, block: bool = True, timeout: int = 5) -> None:
        """原生夹爪持续夹紧。"""
        self._call_void("rm_set_gripper_pick_on", self._validate_range("speed", speed, 1, 1000), self._validate_range("force", force, 50, 1000), bool(block), int(timeout))

    def gripper_move_to(self, position: int, *, block: bool = True, timeout: int = 5) -> None:
        """原生夹爪移动到指定位置。"""
        self._call_void("rm_set_gripper_position", self._validate_range("position", position, 1, 1000), bool(block), int(timeout))

    def get_gripper_state(self) -> GripperState:
        """读取原生夹爪状态。"""
        data = self._call_get("rm_get_gripper_state")
        if not isinstance(data, dict):
            raise RuntimeError("unexpected rm_get_gripper_state payload: %r" % (data,))
        return GripperState(
            enable_state=int(data.get("enable_state", 0)),
            status=int(data.get("status", 0)),
            error=int(data.get("error", 0)),
            mode=int(data.get("mode", 0)),
            current_force=int(data.get("current_force", 0)),
            temperature=int(data.get("temperature", 0)),
            actpos=int(data.get("actpos", 0)),
            raw=data,
        )

    # ------------------------------------------------------------------
    # 末端力传感器
    # ------------------------------------------------------------------
    def get_force_data(self) -> ForceSensorData:
        """
        读取六维力传感器数据。

        返回字段顺序：
        [Fx, Fy, Fz, Mx, My, Mz]

        力单位为 N，力矩单位为 Nm。若当前机械臂不是六维力版本，底层
        SDK 通常会返回错误码。
        """
        data = self._call_get("rm_get_force_data")
        if not isinstance(data, dict):
            raise RuntimeError("unexpected rm_get_force_data payload: %r" % (data,))
        return ForceSensorData(
            force_data=self._as_float_list(data.get("force_data", []), length=6),
            zero_force_data=self._as_float_list(data.get("zero_force_data", []), length=6),
            work_zero_force_data=self._as_float_list(data.get("work_zero_force_data", []), length=6),
            tool_zero_force_data=self._as_float_list(data.get("tool_zero_force_data", []), length=6),
            raw=data,
        )

    def clear_force_data(self) -> None:
        """
        六维力清零。

        会把当前安装姿态、当前负载、当前接触状态作为零点偏置。做超声
        接触力控制时，通常应在探头悬空、未接触体模时调用。
        """
        self._call_void("rm_clear_force_data")

    def get_fz(self) -> FzSensorData:
        """
        读取一维 Z 向力传感器数据。

        力单位为 N。若当前机械臂不是一维力版本，底层 SDK 通常会返回错误码。
        """
        data = self._call_get("rm_get_fz")
        if not isinstance(data, dict):
            raise RuntimeError("unexpected rm_get_fz payload: %r" % (data,))
        return FzSensorData(
            fz=float(data.get("Fz", 0.0)),
            zero_fz=float(data.get("zero_Fz", 0.0)),
            work_zero_fz=float(data.get("work_zero_Fz", 0.0)),
            tool_zero_fz=float(data.get("tool_zero_Fz", 0.0)),
            raw=data,
        )

    def clear_fz(self) -> None:
        """
        一维力清零。

        会把当前 Z 向力作为零点偏置。做接触力判断前，建议在探头悬空、
        未接触体模时调用一次。
        """
        self._call_void("rm_clear_fz")

    def enable_force_position(
        self,
        *,
        sensor: Union[str, int] = "six",
        frame: Union[str, int] = "tool",
        axis: Union[str, int] = "z",
        force: Union[int, float] = 5.0,
    ) -> None:
        """开启单方向原生力位混合控制。

        其余方向继续由后续笛卡尔轨迹控制。该接口与普通 ``movel`` 配套；
        ``force`` 是带方向符号的目标力，单位为 N。正负号决定沿所选轴的
        受力方向；新版周期透传力位控制另有独立接口。
        """
        sensor_map = {"fz": 0, "one": 0, "six": 1, "six_axis": 1}
        frame_map = {"base": 0, "work": 0, "tool": 1, "tcp": 1}
        axis_map = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}

        if isinstance(sensor, str):
            sensor_value = sensor_map.get(sensor.strip().lower())
            if sensor_value is None:
                raise ValueError('sensor must be 0/1, "fz", or "six"')
        else:
            sensor_value = int(sensor)
        if sensor_value not in (0, 1):
            raise ValueError("sensor must be 0(fz) or 1(six)")

        if isinstance(frame, str):
            frame_value = frame_map.get(frame.strip().lower())
            if frame_value is None:
                raise ValueError('frame must be 0/1, "base", or "tool"')
        else:
            frame_value = int(frame)
        if frame_value not in (0, 1):
            raise ValueError("frame must be 0(base) or 1(tool)")

        if isinstance(axis, str):
            axis_value = axis_map.get(axis.strip().lower())
            if axis_value is None:
                raise ValueError('axis must be one of: "x", "y", "z", "rx", "ry", "rz"')
        else:
            axis_value = int(axis)
        if axis_value not in range(6):
            raise ValueError("axis must be in [0, 5]")

        force_value = float(force)
        if not math.isfinite(force_value) or force_value == 0:
            raise ValueError("force must be a finite non-zero value")
        self._call_void("rm_set_force_position", sensor_value, frame_value, axis_value, force_value)

    def disable_force_position(self) -> None:
        """结束原生力位混合控制。"""
        self._call_void("rm_stop_force_position")

    # ------------------------------------------------------------------
    # 示教 / 步进
    # ------------------------------------------------------------------
    def start_drag_teach(self, trajectory_record: Union[bool, int] = False) -> None:
        """
        开始拖动示教。

        参数：
        - trajectory_record : 是否记录拖动轨迹。False/0 表示只进入拖动模式；
          True/1 表示同时记录轨迹，后续可用于轨迹复现。
        """
        record = int(trajectory_record)
        if record not in (0, 1):
            raise ValueError("trajectory_record must be False/0 or True/1")
        self._call_void("rm_start_drag_teach", record)

    def stop_drag_teach(self) -> None:
        """结束拖动示教。"""
        self._call_void("rm_stop_drag_teach")

    def set_force_drag_mode(self, mode: Union[str, int]) -> None:
        """
        设置六维力拖动示教模式。

        mode 支持：
        - 0 / "fast" / "quick" : 快速拖动模式
        - 1 / "precise" / "precision" : 精准拖动模式
        """
        if isinstance(mode, str):
            mode_value = mode.strip().lower()
            if mode_value in ("fast", "quick", "0"):
                mode_int = 0
            elif mode_value in ("precise", "precision", "accurate", "1"):
                mode_int = 1
            else:
                raise ValueError('mode must be 0/1, "fast", or "precise"')
        else:
            mode_int = int(mode)

        if mode_int not in (0, 1):
            raise ValueError("mode must be 0 or 1")
        self._call_void("rm_set_force_drag_mode", mode_int)

    def set_teach_frame(self, frame: Union[str, int]) -> None:
        """设置示教/步进运动参考坐标系：work=当前工作坐标系，tool=当前工具坐标系。"""
        if isinstance(frame, str):
            frame_value = frame.strip().lower()
            if frame_value in ("work", "workspace", "0"):
                frame_int = 0
            elif frame_value in ("tool", "tcp", "1"):
                frame_int = 1
            else:
                raise ValueError('frame must be 0/1, "work", or "tool"')
        else:
            frame_int = int(frame)

        if frame_int not in (0, 1):
            raise ValueError("frame must be 0(work) or 1(tool)")
        self._call_void("rm_set_teach_frame", frame_int)

    def get_teach_frame(self) -> str:
        """获取示教/步进运动参考坐标系，返回 ``work`` 或 ``tool``。"""
        frame_int = int(self._call_get("rm_get_teach_frame"))
        if frame_int == 0:
            return "work"
        if frame_int == 1:
            return "tool"
        raise RuntimeError("unexpected teach frame: %r" % (frame_int,))

    def stop_teach(self) -> None:
        """停止当前连续示教/Jog运动。"""
        self._call_void("rm_set_stop_teach")

    def teach_joint(self, joint_num: int, direction: int, *, v: Union[int, float] = 10) -> None:
        """按关节方向示教。"""
        if int(joint_num) < 1:
            raise ValueError("joint_num must start from 1")
        if int(direction) not in (0, 1):
            raise ValueError("direction must be 0 or 1")
        speed = self._validate_percent("v", v, 1.0, 100.0)
        self._call_void("rm_set_joint_teach", int(joint_num), int(direction), speed)

    def teach_position(self, axis: str, direction: int, *, v: Union[int, float] = 10) -> None:
        """按笛卡尔位置方向示教。"""
        vendor = self._load_vendor_module()
        enum_cls = vendor.rm_pos_teach_type_e
        axis_value = axis.strip().lower()
        if axis_value == "x":
            teach_type = enum_cls.RM_X_DIR_E
        elif axis_value == "y":
            teach_type = enum_cls.RM_Y_DIR_E
        elif axis_value == "z":
            teach_type = enum_cls.RM_Z_DIR_E
        else:
            raise ValueError('axis must be one of: "x", "y", "z"')
        if int(direction) not in (0, 1):
            raise ValueError("direction must be 0 or 1")
        speed = self._validate_percent("v", v, 1.0, 100.0)
        self._call_void("rm_set_pos_teach", teach_type, int(direction), speed)

    def teach_orientation(self, axis: str, direction: int, *, v: Union[int, float] = 10) -> None:
        """按姿态方向示教。"""
        vendor = self._load_vendor_module()
        enum_cls = vendor.rm_ort_teach_type_e
        axis_value = axis.strip().lower()
        if axis_value == "rx":
            teach_type = enum_cls.RM_RX_ROTATE_E
        elif axis_value == "ry":
            teach_type = enum_cls.RM_RY_ROTATE_E
        elif axis_value == "rz":
            teach_type = enum_cls.RM_RZ_ROTATE_E
        else:
            raise ValueError('axis must be one of: "rx", "ry", "rz"')
        if int(direction) not in (0, 1):
            raise ValueError("direction must be 0 or 1")
        speed = self._validate_percent("v", v, 1.0, 100.0)
        self._call_void("rm_set_ort_teach", teach_type, int(direction), speed)

    def step_joint(self, joint_num: int, step: float, *, v: Union[int, float] = 10, block: bool = True) -> None:
        """关节步进。"""
        speed = self._validate_percent("v", v, 1.0, 100.0)
        self._call_void("rm_set_joint_step", int(joint_num), float(step), speed, 1 if block else 0)

    def step_position(self, axis: str, step: float, *, v: Union[int, float] = 10, block: bool = True) -> None:
        """笛卡尔位置步进。"""
        vendor = self._load_vendor_module()
        enum_cls = vendor.rm_pos_teach_type_e
        axis_value = axis.strip().lower()
        if axis_value == "x":
            teach_type = enum_cls.RM_X_DIR_E
        elif axis_value == "y":
            teach_type = enum_cls.RM_Y_DIR_E
        elif axis_value == "z":
            teach_type = enum_cls.RM_Z_DIR_E
        else:
            raise ValueError('axis must be one of: "x", "y", "z"')
        speed = self._validate_percent("v", v, 1.0, 100.0)
        self._call_void("rm_set_pos_step", teach_type, float(step), speed, 1 if block else 0)

    def step_orientation(self, axis: str, step: float, *, v: Union[int, float] = 10, block: bool = True) -> None:
        """姿态步进。"""
        vendor = self._load_vendor_module()
        enum_cls = vendor.rm_ort_teach_type_e
        axis_value = axis.strip().lower()
        if axis_value == "rx":
            teach_type = enum_cls.RM_RX_ROTATE_E
        elif axis_value == "ry":
            teach_type = enum_cls.RM_RY_ROTATE_E
        elif axis_value == "rz":
            teach_type = enum_cls.RM_RZ_ROTATE_E
        else:
            raise ValueError('axis must be one of: "rx", "ry", "rz"')
        speed = self._validate_percent("v", v, 1.0, 100.0)
        self._call_void("rm_set_ort_step", teach_type, float(step), speed, 1 if block else 0)

    # ------------------------------------------------------------------
    # 工具端电源 / Modbus RTU
    # ------------------------------------------------------------------
    def set_tool_voltage(self, voltage_type: int) -> None:
        """
        设置工具端电压。

        常见值：
        - 0 : 关闭输出
        - 2 : 12V
        - 3 : 24V
        """
        if int(voltage_type) not in (0, 2, 3):
            raise ValueError("voltage_type must be 0, 2 or 3")
        self._call_void("rm_set_tool_voltage", int(voltage_type))

    def get_tool_voltage(self) -> int:
        """获取当前工具端电压配置。"""
        return int(self._call_get("rm_get_tool_voltage"))

    def set_modbus_mode(self, *, port: int = 1, baudrate: int = 115200, timeout: int = 3) -> None:
        """
        设置 Modbus RTU 模式。

        参数：
        - port      : 使用哪个 RS485 口
        - baudrate  : 波特率
        - timeout   : 通讯超时
        """
        if int(port) not in (0, 1, 2):
            raise ValueError("port must be 0, 1 or 2")
        if int(baudrate) not in (9600, 115200, 460800):
            raise ValueError("baudrate must be 9600, 115200 or 460800")
        if int(timeout) <= 0:
            raise ValueError("timeout must be > 0")
        self._call_void("rm_set_modbus_mode", int(port), int(baudrate), int(timeout))

    def get_controller_rs485_mode(self) -> Dict[str, Any]:
        """获取控制器 RS485 模式配置。"""
        data = self._call_get("rm_get_controller_rs485_mode")
        if not isinstance(data, dict):
            raise RuntimeError("unexpected rm_get_controller_rs485_mode payload: %r" % (data,))
        return data

    def get_tool_rs485_mode(self) -> Dict[str, Any]:
        """获取工具端 RS485 模式配置。"""
        data = self._call_get("rm_get_tool_rs485_mode")
        if not isinstance(data, dict):
            raise RuntimeError("unexpected rm_get_tool_rs485_mode payload: %r" % (data,))
        return data

    def write_single_register(self, port: int, address: int, value: int, *, device: int = 1) -> None:
        """写单个保持寄存器。"""
        params = self._make_modbus_param(port, address, device)
        self._call_void("rm_write_single_register", params, int(value))

    def write_registers(self, port: int, address: int, values: Sequence[int], *, device: int = 1) -> None:
        """
        连续写多个保持寄存器。

        上层传入 16 位寄存器值；API2 底层要求的是字节数组，所以这里统一
        按 Modbus 大端顺序打包为 high byte / low byte。
        """
        packed: List[int] = []
        for value in values:
            word = int(value) & 0xFFFF
            packed.extend([(word >> 8) & 0xFF, word & 0xFF])
        params = self._make_modbus_param(port, address, device, len(values))
        self._call_void("rm_write_registers", params, packed)

    def read_holding_register(self, port: int, address: int, *, device: int = 1) -> int:
        """读取单个保持寄存器。"""
        params = self._make_modbus_param(port, address, device)
        return int(self._call_get("rm_read_holding_registers", params))

    def read_input_register(self, port: int, address: int, *, device: int = 1) -> int:
        """读取单个输入寄存器。"""
        params = self._make_modbus_param(port, address, device)
        return int(self._call_get("rm_read_input_registers", params))

    def read_multiple_holding_registers(self, port: int, address: int, count: int, *, device: int = 1) -> List[int]:
        """连续读取多个保持寄存器。"""
        if int(count) <= 0:
            raise ValueError("count must be > 0")
        if int(count) == 1:
            return [self.read_holding_register(port, address, device=device)]
        params = self._make_modbus_param(port, address, device, count)
        values = self._call_get("rm_read_multiple_holding_registers", params)
        return [int(v) for v in values]

    def read_multiple_input_registers(
        self,
        port: int,
        address: int,
        count: int,
        *,
        device: int = 1,
    ) -> List[int]:
        """
        连续读取多个输入寄存器。

        注意：
        - API2 返回的是“原始字节列表”，不是“寄存器整数列表”
        - 例如读取 4 个寄存器，通常返回 8 个字节
        - 每 2 个字节需要再拼成 1 个 16 位寄存器
        """
        if int(count) <= 0:
            raise ValueError("count must be > 0")

        arm = self._require_arm()
        if hasattr(arm, "rm_read_multiple_input_registers"):
            params = self._make_modbus_param(port, address, device, count)
            values = self._call_get("rm_read_multiple_input_registers", params)
            return [int(v) & 0xFF for v in values]

        raise AttributeError("API2 SDK missing method: rm_read_multiple_input_registers")

    # ------------------------------------------------------------------
    # 大寰夹爪阻塞辅助
    # ------------------------------------------------------------------
    def _wait_dh_initialized(
        self,
        *,
        port: int,
        device: int,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
    ) -> None:
        """
        等待大寰夹爪初始化完成。

        判定条件：
        - init_status == 1
        """
        end_time = time.time() + float(timeout)

        while time.time() < end_time:
            state = self.get_gripper_status_dh(port=port, device=device)
            if state.init_status == 1:
                return
            time.sleep(float(poll_interval))

        raise TimeoutError("DH gripper init timeout")

    def _wait_dh_motion_done(
        self,
        *,
        port: int,
        device: int,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
    ) -> None:
        """
        等待大寰夹爪动作完成。

        判定逻辑：
        - grip_status != 0
        即不再处于“运动中”
        """
        end_time = time.time() + float(timeout)

        while time.time() < end_time:
            state = self.get_gripper_status_dh(port=port, device=device)
            if state.grip_status != 0:
                return
            time.sleep(float(poll_interval))

        raise TimeoutError("DH gripper motion timeout")

    # ------------------------------------------------------------------
    # 大寰夹爪
    # ------------------------------------------------------------------
    def control_gripper_dh(
        self,
        action: str,
        * ,
        speed: int = 50,
        force: int = 50,
        position: int = 0,
        port: int = 1,
        device: int = 1,
        baudrate: int = 115200,
        auto_prepare: bool = False,
        tool_voltage_type: int = 3,
        block: bool = False,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
    ) -> None:
        """
        控制大寰夹爪。

        设计思路：
        - 一个控制接口，负责下发动作命令
        - 一个状态接口，负责读取状态
        - 现在额外增加 block 参数：
            block=False : 只发命令，不等待
            block=True  : 发命令后轮询状态，直到动作完成或超时

        参数：
        - action:
            init / open / close / pose / move / position
        - auto_prepare:
            为 True 时自动做：
            1. 工具端上电
            2. 配置 Modbus
            3. 关闭 IO 模式
        - block:
            是否阻塞等待动作完成
        - timeout:
            阻塞等待的最大时长
        - poll_interval:
            阻塞等待时轮询状态的间隔
        """
        action = action.strip().lower()

        if auto_prepare:
            self.init_external_gripper(
                port=port,
                baudrate=baudrate,
                timeout=3,
                tool_voltage_type=tool_voltage_type,
                switch_to_modbus=True,
            )
            # 大寰常用配置里，需要关闭 IO 模式，改为 Modbus 控制
            self.write_single_register(port, self._DH_REG_IO_MODE_SWITCH, 0, device=device)

        if action == "init":
            self.write_single_register(port, self._DH_REG_INIT_CMD, 0xA5, device=device)
            if block:
                self._wait_dh_initialized(
                    port=port,
                    device=device,
                    timeout=timeout,
                    poll_interval=poll_interval,
                )
            return

        speed = self._validate_range("speed", speed, 1, 100)
        force = self._validate_range("force", force, 20, 100)
        position = self._validate_range("position", position, 0, 1000)

        # 大寰协议一般先写力值和速度，再写目标位置
        self.write_single_register(port, self._DH_REG_FORCE, force, device=device)
        self.write_single_register(port, self._DH_REG_SPEED, speed, device=device)

        if action == "open":
            target_position = 1000
        elif action == "close":
            target_position = 0
        elif action in ("pose", "move", "position"):
            target_position = position
        else:
            raise ValueError("action must be one of: init, open, close, pose, move, position")

        self.write_single_register(port, self._DH_REG_TARGET_POSITION, target_position, device=device)

        if block:
            self._wait_dh_motion_done(
                port=port,
                device=device,
                timeout=timeout,
                poll_interval=poll_interval,
            )

    def get_gripper_status_dh(self, *, port: int = 1, device: int = 1) -> DHGripperState:
        """读取大寰夹爪状态。"""
        return DHGripperState(
            init_status=self.read_holding_register(port, self._DH_REG_INIT_STATE, device=device),
            grip_status=self.read_holding_register(port, self._DH_REG_GRIP_STATE, device=device),
            current_position=self.read_holding_register(port, self._DH_REG_CURRENT_POSITION, device=device),
        )

    # ------------------------------------------------------------------
    # 钧舵夹爪阻塞辅助
    # ------------------------------------------------------------------
    def _wait_jd_initialized(
        self,
        *,
        port: int,
        device: int,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
    ) -> None:
        """
        等待钧舵夹爪激活完成。

        判定条件：
        - state.activated == True
        """
        end_time = time.time() + float(timeout)

        while time.time() < end_time:
            state = self.get_gripper_status_jd(port=port, device=device)
            if state.activated:
                return
            time.sleep(float(poll_interval))

        raise TimeoutError("JD gripper init timeout")

    def _wait_jd_motion_done(
        self,
        *,
        target_position: int,
        port: int,
        device: int,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
        position_tolerance: int = 3,
    ) -> None:
        """
        等待钧舵夹爪动作完成。

        判定逻辑（尽量简单而实用）：
        满足以下任一条件就认为动作结束：
        1. 当前位置已经接近目标位置
        2. 已经夹住物体
        3. 已经掉物 / 到位未检测到物体
        """
        end_time = time.time() + float(timeout)

        while time.time() < end_time:
            state = self.get_gripper_status_jd(port=port, device=device)

            # 条件 1：位置到位
            if abs(int(state.current_position) - int(target_position)) <= int(position_tolerance):
                return

            # 条件 2：夹住物体
            if state.holding:
                return

            # 条件 3：发生掉物或其他到位结果
            if state.dropped:
                return

            time.sleep(float(poll_interval))

        raise TimeoutError("JD gripper motion timeout")

    # ------------------------------------------------------------------
    # 钧舵夹爪
    # ------------------------------------------------------------------
    def control_gripper_jd(
        self,
        action: str,
        * ,
        position: int = 0,
        speed: int = 128,
        force: int = 128,
        port: int = 1,
        device: int = _JD_DEFAULT_DEVICE,
        baudrate: int = 115200,
        auto_prepare: bool = False,
        tool_voltage_type: int = 3,
        block: bool = False,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
        position_tolerance: int = 3,
    ) -> None:
        """
        控制钧舵夹爪。

        设计思路：
        - 一个控制接口
        - 一个状态读取接口
        - 现在增加 block 参数，实现“可选阻塞等待”

        参数：
        - action:
            init / open / close / pose / move / position
        - auto_prepare:
            True 时自动做：
            1. 工具端上电
            2. 配置末端 Modbus
        - block:
            是否阻塞等待动作完成
        - timeout:
            最大等待时间
        - poll_interval:
            轮询状态的时间间隔
        - position_tolerance:
            位置到位容差
        """
        action = action.strip().lower()

        if auto_prepare:
            self.init_external_gripper(
                port=port,
                baudrate=baudrate,
                timeout=3,
                tool_voltage_type=tool_voltage_type,
                switch_to_modbus=True,
            )

        if action == "init":
            # 常见初始化方式：
            # 先清使能，再置使能
            # 钧舵官方 SDK 使用 0x10（写多个保持寄存器），即便只写一个寄存器也是如此。
            self.write_registers(port, self._JD_REG_CTRL, [0x0000], device=device)
            self.write_registers(port, self._JD_REG_CTRL, [0x0001], device=device)

            if block:
                self._wait_jd_initialized(
                    port=port,
                    device=device,
                    timeout=timeout,
                    poll_interval=poll_interval,
                )
            return

        position = self._validate_range("position", position, 0, 255)
        speed = self._validate_range("speed", speed, 0, 255)
        force = self._validate_range("force", force, 0, 255)

        if action == "open":
            target_position = 0
        elif action == "close":
            target_position = 255
        elif action in ("pose", "move", "position"):
            target_position = position
        else:
            raise ValueError("action must be one of: init, open, close, pose, move, position")

        # 钧舵官方 SDK 从 0x03E8 连续写 3 个寄存器：
        # [控制字, 目标位置, 速度/力]。这样能保证参数和触发命令在同一帧中生效。
        self.write_registers(
            port,
            self._JD_REG_CTRL,
            [
                0x0009,
                (target_position & 0xFF) << 8,
                ((force & 0xFF) << 8) | (speed & 0xFF),
            ],
            device=device,
        )

        if block:
            self._wait_jd_motion_done(
                target_position=target_position,
                port=port,
                device=device,
                timeout=timeout,
                poll_interval=poll_interval,
                position_tolerance=position_tolerance,
            )

    def get_gripper_status_jd(
        self,
        *,
        port: int = 1,
        device: int = _JD_DEFAULT_DEVICE,
    ) -> JDGripperState:
        """
        读取钧舵夹爪状态。

        读取 0x07D0 ~ 0x07D3 共 4 个输入寄存器。
        API2 返回的是 8 个字节，需要两两拼成 4 个 16 位寄存器。
        """
        data = self.read_multiple_input_registers(
            port,
            self._JD_REG_STATUS,
            4,
            device=device,
        )

        if len(data) != 8:
            raise RuntimeError("unexpected JD byte length: %r" % (data,))

        # 每 2 个字节拼成 1 个 16 位寄存器（高字节在前，低字节在后）
        reg_status = ((data[0] & 0xFF) << 8) | (data[1] & 0xFF)
        reg_fault_pos = ((data[2] & 0xFF) << 8) | (data[3] & 0xFF)
        reg_speed_force = ((data[4] & 0xFF) << 8) | (data[5] & 0xFF)
        reg_volt_temp = ((data[6] & 0xFF) << 8) | (data[7] & 0xFF)

        # 按钧舵寄存器协议拆字段
        status_byte = reg_status & 0xFF
        fault_byte = reg_fault_pos & 0xFF
        pos_byte = (reg_fault_pos >> 8) & 0xFF
        speed_byte = reg_speed_force & 0xFF
        force_byte = (reg_speed_force >> 8) & 0xFF
        volt_byte = reg_volt_temp & 0xFF
        temp_byte = (reg_volt_temp >> 8) & 0xFF

        return JDGripperState(
            gact=status_byte & 0x01,
            gdrop_sta=(status_byte >> 1) & 0x01,
            gmode=(status_byte >> 2) & 0x01,
            ggto=(status_byte >> 3) & 0x01,
            gsta=(status_byte >> 4) & 0x03,
            gobj=(status_byte >> 6) & 0x03,
            fault_code=fault_byte,
            current_position=pos_byte,
            current_speed=speed_byte,
            current_force=force_byte,
            bus_voltage=volt_byte,
            temperature=self._to_signed_8bit(temp_byte),
        )

    # ------------------------------------------------------------------
    # 升降柱
    # ------------------------------------------------------------------
    def control_lift(
        self,
        action: str,
        *,
        speed: int = 30,
        height: int = 0,
        block: bool = True,
    ) -> None:
        """
        控制升降柱。

        支持动作：
        - up   : 上升
        - down : 下降
        - stop : 停止
        - to   : 到指定高度
        """
        action = action.strip().lower()

        if action == "stop":
            self._call_void("rm_set_lift_speed", 0)
            return

        if action == "up":
            self._call_void("rm_set_lift_speed", self._validate_range("speed", speed, 1, 100))
            return

        if action == "down":
            self._call_void("rm_set_lift_speed", -self._validate_range("speed", speed, 1, 100))
            return

        if action == "to":
            speed_value = self._validate_range("speed", speed, 1, 100)
            if int(height) < 0:
                raise ValueError("height must be >= 0")
            self._call_void("rm_set_lift_height", speed_value, int(height), 1 if block else 0)
            return

        raise ValueError("action must be one of: up, down, stop, to")

    def get_lift_status(self) -> LiftState:
        """获取升降柱状态。"""
        data = self._call_get("rm_get_lift_state")
        if not isinstance(data, dict):
            raise RuntimeError("unexpected rm_get_lift_state payload: %r" % (data,))
        return LiftState(
            height=int(data.get("height", 0)),
            current=int(data.get("current", 0)),
            err=int(data.get("err", 0)),
            mode=int(data.get("mode", 0)),
            raw=data,
        )

    # ------------------------------------------------------------------
    # UDP 主动上报
    # ------------------------------------------------------------------
    def set_realtime_push(
        self,
        *,
        cycle: int = 100,
        port: int = 8089,
        enable: bool = True,
        force_coordinate: int = 0,
        ip: str = "",
        joint_speed: int = 0,
        lift_state: int = 0,
        expand_state: int = 0,
    ) -> None:
        """
        设置 UDP 主动上报配置。
        """
        vendor = self._load_vendor_module()
        custom = vendor.rm_udp_custom_config_t()
        custom.joint_speed = int(joint_speed)
        custom.lift_state = int(lift_state)
        custom.expand_state = int(expand_state)
        config = vendor.rm_realtime_push_config_t(int(cycle), bool(enable), int(port), int(force_coordinate), str(ip), custom)
        self._call_void("rm_set_realtime_push", config)

    def get_realtime_push(self) -> Dict[str, Any]:
        """获取 UDP 主动上报配置。"""
        data = self._call_get("rm_get_realtime_push")
        if not isinstance(data, dict):
            raise RuntimeError("unexpected rm_get_realtime_push payload: %r" % (data,))
        return data

    def start_realtime_listener(self, callback: Callable[[Any], None]) -> None:
        """注册实时状态回调。"""
        self.raw_call("rm_realtime_arm_state_call_back", callback, check=False)


__all__ = [
    "ArmApiError",
    "ArmPose",
    "ArmState",
    "ControllerState",
    "GripperState",
    "DHGripperState",
    "JDGripperState",
    "LiftState",
    "RealmanArmClient",
]
