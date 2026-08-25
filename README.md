# PICO + RealMan 双臂高跟随遥操作

本仓库基于 OpenDriveLab/TAMEn 的 tAmeR PICO 遥操作链路，将原有机械臂后端替换为已在两台 RealMan RM65 上验证的双臂 CANFD 高跟随控制，并包含三台 Intel RealSense D435 的 WebSocket-JPEG 视频服务。

控制链路：

```text
PICO TCP 8018
  -> ROS 2 数据分发
  -> 手柄锚点和水平朝向补偿
  -> 左右臂独立坐标映射
  -> 笛卡尔速度/加速度限幅（125 Hz）
  -> RealMan movep_canfd(follow=True, trajectory_mode=0)
```

当前实机参数：

- 左臂 IP：`169.254.128.18`
- 右臂 IP：`169.254.128.19`
- 控制电脑有线地址：`169.254.128.20`
- 左臂位置/姿态映射：`z,-y,-x` / `-z,y,-x`
- 右臂位置/姿态映射：`z,y,x` / `-z,y,x`
- 位置比例：`0.40`
- 姿态比例：`0.05`
- CANFD 发送频率：`125 Hz`
- 平移速度上限：`150 mm/s`
- 平移加速度上限：`1200 mm/s^2`

## 目录

```text
camera_api/                  RealSense 视频 WebSocket 服务
pico_app/                    tAmeR PICO APP 版本与本地安装说明（APK 不入库）
src/realman_teleop/          当前实机使用的 RealMan CANFD、映射与安全核心
src/vr_data_pub/             PICO TCP 8018 接收与 ROS 2 数据分发
requirements-camera.txt      摄像头 Python 依赖
```

## 上游来源

本项目复用了 [OpenDriveLab/TAMEn](https://github.com/OpenDriveLab/TAMEn) 发布的 tAmeR PICO 应用、TCP 数据格式和 ROS 2 接收思路。主要改动包括：

- 将 TAMEn 的 JAKA 控制后端替换为 RealMan API2 和 `movep_canfd` 高跟随控制；
- 增加双臂独立坐标映射、水平朝向补偿、锚点保护、watchdog 和急停；
- 增加三台 RealSense D435 的直接 WebSocket-JPEG 视频服务；
- 为当前实验网络制作 tAmeR 视频端点修改版 APK。

上游项目及其对象形式发布物采用 Apache License 2.0。本仓库的详细归属和修改声明见 [NOTICE](NOTICE)。使用本项目开展研究时，也请引用 TAMEn 原论文，引用信息见上游 README。

## 环境

已验证环境：

- Ubuntu 22.04，aarch64
- ROS 2 Humble
- Python 3.10
- RealMan API2
- Intel RealSense D435 x3

安装 ROS 侧依赖：

```bash
sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-numpy \
  python3-scipy \
  ros-humble-geometry-msgs \
  ros-humble-sensor-msgs \
  ros-humble-std-msgs \
  ros-humble-std-srvs
```

从 RealMan 官方发布安装 API2 Python 包：

```bash
python3 -m pip install Robotic_Arm
python3 -c "from Robotic_Arm.rm_robot_interface import RoboticArm; print('RealMan API2 OK')"
```

官方源码和不同平台的 SDK 位于：

```text
https://github.com/RealManRobot/RM_API2
```

摄像头依赖可以安装到现有的 `robot` Conda 环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robot
python3 -m pip install -r requirements-camera.txt
```

## 构建

在仓库根目录执行：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select vr_data_pub realman_teleop
source install/setup.bash
```

每次打开新终端都需要重新执行：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

## 网络检查

机械臂使用有线网络，控制电脑网口应配置为 `169.254.128.20/24`。启动前检查：

```bash
ping -c 3 169.254.128.18
ping -c 3 169.254.128.19
```

PICO 和控制电脑使用同一 Wi-Fi。PICO APP 中的控制服务器地址应设置为控制电脑 Wi-Fi IP，端口为 `8018`。

## 安装 PICO APP

APK 文件较大且使用本地开发证书签名，因此不纳入 Git 仓库。当前实验使用的本地文件名为：

```text
pico_app/tAmeR_192.168.3.6_video_v2.apk
```

将 APK 单独放入上述路径后再执行安装。该 APK 是 TAMEn `tAmeR.apk` 的本地修改构建，视频 WebSocket 端点适配当前控制电脑 Wi-Fi 地址 `192.168.3.6:8765`。它不是 PICO 商店应用，也不是 OpenDriveLab 发布的原始签名版本。详细版本、签名和校验信息见 [pico_app/README.md](pico_app/README.md)。需要上游原版时，请从 [OpenDriveLab/TAMEn](https://github.com/OpenDriveLab/TAMEn) 获取。

在已配置 ADB 的电脑上连接 PICO 后安装：

```bash
adb install -r pico_app/tAmeR_192.168.3.6_video_v2.apk
```

如果设备中已经安装了不同签名的 `com.TAMEn.tAmeR`，Android 会拒绝覆盖。确认不需要保留旧应用数据后执行：

```bash
adb uninstall com.TAMEn.tAmeR
adb install pico_app/tAmeR_192.168.3.6_video_v2.apk
```

安装后，在 PICO APP 中将控制服务器设置为控制电脑 Wi-Fi IP，端口设置为 `8018`。如果控制电脑不再使用 `192.168.3.6`，当前修改版的视频地址也需要相应调整；也可改用 TAMEn 上游发布的原版 APK。

## 启动机械臂

### 双臂

终端 A：

```bash
cd /path/to/realman_pico_teleop
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_LOG_DIR=/tmp/realman-ros-log

ros2 launch realman_teleop \
  realman_tamer_dual_position_live_check.launch.py \
  live_confirmation:=I_UNDERSTAND_REALMAN_LIVE_CONTROL
```

### 仅左臂

```bash
ros2 launch realman_teleop \
  realman_tamer_left_position_live_check.launch.py \
  live_confirmation:=I_UNDERSTAND_REALMAN_LIVE_CONTROL
```

正常启动时应看到以下关键信息：

```text
follow=True
command_rate=125.0Hz
target_path=scaled-accel-limited
cartesian_limits=150mm/s/1200mm/s2
trajectory=0/0
```

手柄控制：

- 按住 `LT`：左臂位置控制；松开停止。
- 按住 `LG`：左臂姿态控制；松开停止。
- 按住 `RT`：右臂位置控制；松开停止。
- 按住 `RG`：右臂姿态控制；松开停止。

按下 LT/LG/RT/RG 时会捕获当前位置和水平朝向锚点。操作者改变站立朝向后，应松开并重新按下对应按键。面对、背对或与机器人呈 `+/-90` 度时，前后/左右映射保持一致。

## 急停

左臂：

```bash
ros2 service call /left_arm/emergency_stop \
  std_srvs/srv/SetBool "{data: true}"
```

右臂：

```bash
ros2 service call /right_arm/emergency_stop \
  std_srvs/srv/SetBool "{data: true}"
```

发生故障后，在确认机械臂状态正常的情况下清除软件故障锁：

```bash
ros2 service call /left_arm/clear_fault std_srvs/srv/SetBool "{data: true}"
ros2 service call /right_arm/clear_fault std_srvs/srv/SetBool "{data: true}"
```

## 启动三台摄像头

当前相机顺序：

| 顺序 | 序列号 | 安装位置 |
|---|---|---|
| 1 | `405622072991` | 头部 |
| 2 | `419522072006` | 右手侧 |
| 3 | `420122071063` | 左手侧 |

终端 B：

```bash
cd /path/to/realman_pico_teleop
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robot

python3 camera_api/tamer_realsense_ws.py \
  --serial 405622072991 \
  --serial 419522072006 \
  --serial 420122071063 \
  --host 0.0.0.0 \
  --port 8765 \
  --stream-fps 10 \
  --jpeg-quality 70
```

枚举相机：

```bash
python3 camera_api/camera_list.py
```

健康检查：

```bash
curl http://127.0.0.1:8765/health
```

PICO APP 使用的视频地址为：

```text
ws://<控制电脑Wi-Fi-IP>:8765/ws/video
```

如果健康检查中的 `clients` 为 `0`，表示 APP 尚未连接视频 WebSocket；大于等于 `1` 表示已连接。

## 测试

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon test --packages-select realman_teleop
colcon test-result --verbose
```

## 常见问题

相机出现 `Device or resource busy` 时，先停止旧的相机进程：

```bash
pgrep -af 'realsense|D435|tamer_realsense_ws|python.*camera'
```

PICO 无数据时检查：

```bash
ss -ltnp | grep 8018
ros2 topic hz /vr_raw_data
```

机械臂无响应时，先检查网络、电源状态和示教器报警，不要在错误未清除时反复按下控制键。

## SDK 与许可证

项目代码采用 Apache-2.0 许可证，详见 `LICENSE`。本项目基于 TAMEn 修改，保留上游许可证并在 `NOTICE` 中列出来源和主要变更。

本仓库不包含 RealMan 厂商 SDK 源码或二进制。请通过 `pip install Robotic_Arm` 或 [RealMan 官方 RM_API2 仓库](https://github.com/RealManRobot/RM_API2) 安装与当前操作系统和 CPU 架构匹配的版本。

机械臂 live 控制具有实际运动风险。首次部署、修改映射或更换机械臂后，应降低动作幅度并确保物理急停可立即触达。
