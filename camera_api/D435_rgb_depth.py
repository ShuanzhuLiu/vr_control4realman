# -*- coding: utf-8 -*-
"""
d435_simple_wrapper.py

用途：
1. 枚举当前连接的 RealSense 相机
2. 读取每台相机的序列号、名称、产品线等信息
3. 按“指定序列号”启动你想要的那一台相机
4. 获取彩色图、深度图、深度尺度、相机内参
5. 支持 depth 对齐到 color
"""

import time
from typing import Optional, Dict, List

import cv2
import numpy as np
import pyrealsense2 as rs


def list_realsense_devices() -> List[Dict]:
    """
    枚举当前所有已连接的 RealSense 设备

    返回:
        一个列表，每个元素都是一台相机的信息字典，例如：
        {
            "index": 0,
            "name": "Intel RealSense D435I",
            "serial": "123456789",
            "product_line": "D400",
            "usb_type": "3.2"
        }
    """
    ctx = rs.context()
    devices = []

    for idx, dev in enumerate(ctx.devices):
        # 某些电脑会出现“platform camera”之类的设备，这里过滤掉
        try:
            name = dev.get_info(rs.camera_info.name)
        except Exception:
            name = "Unknown"

        if name.lower() == "platform camera":
            continue

        # 分别读取设备信息
        try:
            serial = dev.get_info(rs.camera_info.serial_number)
        except Exception:
            serial = ""

        try:
            product_line = dev.get_info(rs.camera_info.product_line)
        except Exception:
            product_line = ""

        try:
            usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
        except Exception:
            usb_type = ""

        devices.append({
            "index": idx,
            "name": name,
            "serial": serial,
            "product_line": product_line,
            "usb_type": usb_type,
        })

    return devices


def print_realsense_devices() -> None:
    """
    打印当前所有已连接相机，方便你复制序列号
    """
    devices = list_realsense_devices()

    if not devices:
        print("未检测到任何 RealSense 相机")
        return

    print("检测到以下 RealSense 相机：")
    for dev in devices:
        print(
            f"[{dev['index']}] "
            f"name={dev['name']}, "
            f"serial={dev['serial']}, "
            f"product_line={dev['product_line']}, "
            f"usb={dev['usb_type']}"
        )


class D435Camera:
    """
    一个“只管单台相机”的简化封装类

    设计思路：
    - 初始化时传入 serial（推荐）
    - start() 时只启动这一台
    - get_frames() 获取 color / depth
    - stop() 停止相机

    这样你的多相机使用方式就变成：
    cam1 = D435Camera(serial="左相机序列号")
    cam2 = D435Camera(serial="右相机序列号")

    谁需要启动，就启动谁
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_color: bool = True,
        enable_depth: bool = True,
        align_depth_to_color: bool = True,
    ):
        """
        参数说明：
            serial:
                指定要启动的相机序列号。
                如果你的电脑只插了一台相机，也可以不填；
                但如果有多台，强烈建议填写序列号。

            width / height / fps:
                图像分辨率和帧率

            enable_color:
                是否开启彩色图

            enable_depth:
                是否开启深度图

            align_depth_to_color:
                是否把 depth 对齐到 color
                如果后面你做“彩色图上点一个像素，再取对应深度”，建议开 True
        """
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.align_depth_to_color = align_depth_to_color

        self.pipeline = None
        self.config = None
        self.profile = None
        self.align = None
        self.depth_scale = None

    def start(self) -> None:
        """
        启动相机

        关键点：
        - 如果指定了 serial，就只启动这一台
        - 如果没指定 serial，则默认启动系统分配到的某一台
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 如果指定了序列号，则绑定到对应相机
        if self.serial:
            self.config.enable_device(self.serial)

        # 开启深度流
        if self.enable_depth:
            self.config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps
            )

        # 开启彩色流
        if self.enable_color:
            self.config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps
            )

        # 真正启动
        self.profile = self.pipeline.start(self.config)

        # 获取深度尺度（把 depth 原始值转换成“米”会用到）
        if self.enable_depth:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

        # 如果需要 depth 对齐到 color，则创建 align 对象
        if self.enable_depth and self.enable_color and self.align_depth_to_color:
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None

        # 预热几帧，避免刚启动时图像不稳定
        for _ in range(10):
            try:
                self.pipeline.poll_for_frames()
            except Exception:
                pass
            time.sleep(0.01)

        print(f"相机启动成功，serial={self.get_active_device_serial()}")

    def stop(self) -> None:
        """
        停止相机
        """
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        self.pipeline = None
        self.config = None
        self.profile = None
        self.align = None

    def get_active_device_serial(self) -> str:
        """
        获取当前实际启动的设备序列号
        """
        if self.profile is None:
            return ""

        dev = self.profile.get_device()
        return dev.get_info(rs.camera_info.serial_number)

    def get_active_device_name(self) -> str:
        """
        获取当前实际启动的设备名称
        """
        if self.profile is None:
            return ""

        dev = self.profile.get_device()
        return dev.get_info(rs.camera_info.name)

    def get_frames(self, timeout_ms: int = 3000) -> Dict:
        """
        获取一帧数据

        返回：
            {
                "color": 彩色图(np.ndarray) 或 None,
                "depth": 深度图(np.ndarray) 或 None,
                "timestamp_ms": 时间戳,
                "depth_scale": 深度尺度
            }
        """
        if self.pipeline is None:
            raise RuntimeError("相机尚未启动，请先调用 start()")

        # 等待一帧
        frames = self.pipeline.wait_for_frames(timeout_ms)

        # 如果开启了对齐，则先做 depth->color 对齐
        if self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame() if self.enable_color else None
        depth_frame = frames.get_depth_frame() if self.enable_depth else None

        color_image = None
        depth_image = None
        timestamp_ms = None

        if color_frame:
            color_image = np.asanyarray(color_frame.get_data()).copy()
            timestamp_ms = float(color_frame.get_timestamp())

        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data()).copy()
            if timestamp_ms is None:
                timestamp_ms = float(depth_frame.get_timestamp())

        if timestamp_ms is None:
            timestamp_ms = time.time() * 1000.0

        return {
            "color": color_image,
            "depth": depth_image,
            "timestamp_ms": timestamp_ms,
            "depth_scale": self.depth_scale,
        }

    def get_intrinsics(self, stream_type: str = "color") -> Dict:
        """
        获取相机内参

        参数：
            stream_type: "color" 或 "depth"

        返回：
            {
                "width": ...,
                "height": ...,
                "fx": ...,
                "fy": ...,
                "ppx": ...,
                "ppy": ...,
                "model": ...,
                "coeffs": [...]
            }
        """
        if self.profile is None:
            raise RuntimeError("相机尚未启动，请先调用 start()")

        if stream_type not in ("color", "depth"):
            raise ValueError("stream_type 只能是 'color' 或 'depth'")

        target_stream = rs.stream.color if stream_type == "color" else rs.stream.depth

        for s in self.profile.get_streams():
            if s.stream_type() == target_stream:
                vsp = s.as_video_stream_profile()
                intr = vsp.get_intrinsics()
                return {
                    "width": intr.width,
                    "height": intr.height,
                    "fx": intr.fx,
                    "fy": intr.fy,
                    "ppx": intr.ppx,
                    "ppy": intr.ppy,
                    "model": str(intr.model),
                    "coeffs": list(intr.coeffs),
                }

        raise RuntimeError(f"没有找到 {stream_type} 流的内参")

    def get_distance(self, depth_image: np.ndarray, u: int, v: int) -> float:
        """
        获取深度图某个像素点对应的距离（单位：米）

        参数：
            depth_image: get_frames() 返回的 depth 图
            u, v: 像素坐标

        返回：
            距离（米）
        """
        if depth_image is None:
            raise ValueError("depth_image 为空")

        if self.depth_scale is None:
            raise RuntimeError("depth_scale 不可用，请确认是否开启了 depth")

        # depth 图里的值通常是 uint16 原始值
        raw_depth = depth_image[v, u]
        distance_m = float(raw_depth) * float(self.depth_scale)
        return distance_m

    def pixel_to_point(self, u: int, v: int, depth_m: float) -> List[float]:
        """
        把像素坐标 + 深度，转换成相机坐标系下的 3D 点

        返回：
            [x, y, z]，单位：米
        """
        if self.align_depth_to_color and self.enable_color:
            intr_dict = self.get_intrinsics("color")
        else:
            intr_dict = self.get_intrinsics("depth")

        # 重新构造 intrinsics 对象
        intr = rs.intrinsics()
        intr.width = intr_dict["width"]
        intr.height = intr_dict["height"]
        intr.fx = intr_dict["fx"]
        intr.fy = intr_dict["fy"]
        intr.ppx = intr_dict["ppx"]
        intr.ppy = intr_dict["ppy"]
        intr.model = rs.distortion.brown_conrady
        intr.coeffs = intr_dict["coeffs"]

        point = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(depth_m))
        return [float(point[0]), float(point[1]), float(point[2])]


if __name__ == "__main__":
    """
    演示用法：
    1. 先打印所有相机
    2. 把你想用的那台相机 serial 填进去
    3. 程序只启动那一台
    """

    # 第一步：先看当前插了哪些相机
    print_realsense_devices()

    # 第二步：把这里改成你想启动的那台相机的序列号
    # 比如：target_serial = "243522073236"
    #target_serial = "405622072991"  #头部
    #target_serial = "419522072006"  #右手
    #target_serial = "420122071063"  #左手
    target_serial="346222071235"

    # 如果你电脑上只插了一台相机，也可以先写 None
    # 如果是多台相机，建议一定写 serial，避免启动错设备
    cam = D435Camera(
        serial=target_serial,
        width=640,
        height=480,
        fps=30,
        enable_color=True,
        enable_depth=True,
        align_depth_to_color=True,
    )

    cam.start()

    try:
        while True:
            data = cam.get_frames()

            color = data["color"]
            depth = data["depth"]

            if color is not None:
                cv2.imshow("color", color)

            if depth is not None:
                # 深度图转成便于显示的灰度图
                depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
                cv2.imshow("depth", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        cam.stop()
        E
        E
        cv2.destroyAllWindows()
