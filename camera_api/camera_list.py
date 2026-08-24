import time

try:
    import pyrealsense2 as rs
except ImportError:
    print("[ERROR] 未安装 pyrealsense2，请先安装：pip install pyrealsense2")
    raise SystemExit(1)


def safe_get_info(dev, info_key, default="N/A"):
    try:
        if dev.supports(info_key):
            return dev.get_info(info_key)
    except Exception:
        pass
    return default


def list_realsense_devices(rounds=3, interval=1.0):
    """
    多轮检测 RealSense 设备，避免刚插上时枚举不完整
    """
    best_devices = []

    for r in range(rounds):
        ctx = rs.context()
        devices = ctx.query_devices()

        current_devices = []
        print(f"\n[INFO] 第 {r + 1}/{rounds} 轮检测结果：")

        for i, dev in enumerate(devices):
            info = {
                "index": i,
                "name": safe_get_info(dev, rs.camera_info.name),
                "serial": safe_get_info(dev, rs.camera_info.serial_number),
                "product_line": safe_get_info(dev, rs.camera_info.product_line),
                "usb_type": safe_get_info(dev, rs.camera_info.usb_type_descriptor),
                "physical_port": safe_get_info(dev, rs.camera_info.physical_port),
                "firmware": safe_get_info(dev, rs.camera_info.firmware_version),
            }
            current_devices.append(info)

            print(
                f"  [{info['index']}] "
                f"name={info['name']}, "
                f"serial={info['serial']}, "
                f"product_line={info['product_line']}, "
                f"usb_type={info['usb_type']}, "
                f"physical_port={info['physical_port']}, "
                f"firmware={info['firmware']}"
            )

        print(f"[INFO] 本轮检测到设备数量：{len(current_devices)}")

        if len(current_devices) > len(best_devices):
            best_devices = current_devices

        if r != rounds - 1:
            time.sleep(interval)

    print("\n========== 最终结果 ==========")
    print(f"[INFO] 检测到 RealSense 设备数量：{len(best_devices)}")

    if not best_devices:
        print("[WARN] 没有检测到任何 RealSense 设备")
    else:
        print("[INFO] 序列号列表：")
        for dev in best_devices:
            print(f"  - {dev['serial']}  ({dev['name']})")

    return best_devices


if __name__ == "__main__":
    list_realsense_devices(rounds=5, interval=1.0)