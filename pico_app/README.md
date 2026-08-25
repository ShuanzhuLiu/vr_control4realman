# tAmeR PICO 应用

当前 RealMan 双臂实验使用以下本地 PICO APK：

```text
tAmeR_192.168.3.6_video_v2.apk
```

APK 文件较大且使用本地开发证书签名，因此默认被 `.gitignore` 排除，不随本仓库上传。使用前需通过实验设备备份或其他文件传输方式将其放入本目录。需要上游原版时，请从 [OpenDriveLab/TAMEn](https://github.com/OpenDriveLab/TAMEn) 获取。

它基于 [OpenDriveLab/TAMEn](https://github.com/OpenDriveLab/TAMEn) 发布的 `tAmeR.apk` 修改，并继续按 Apache License 2.0 分发。该构建用于 PICO 4 / PICO 4 Ultra，PICO 与控制电脑必须位于同一局域网。

## 本地修改

- 视频 WebSocket 服务地址适配当前实验电脑 `192.168.3.6:8765`；
- APK 使用本地开发证书重新签名；
- Android 包名保持为 `com.TAMEn.tAmeR`。

本地工作目录只需保留当前实机版本，不需要复制旧 APK 或 `.idsig` 增量安装辅助文件。不同局域网地址下使用时，应调整视频端点或使用 TAMEn 上游原版 APK。

## 完整性与签名

```text
文件大小：65076816 bytes
SHA-256：f67a4670e6a336aea24b5429c69f34687f3947f669bcf2b7264147561d631b37
签名方案：APK Signature Scheme v3
签名证书：CN=tAmeR Video V2, OU=Local Development, O=TAMEn
证书 SHA-256：f240cfd7d285887827418dea5a94e4c4b24a2387a76085d1847b09bdeebe0342
```

下载或复制后可校验：

```bash
sha256sum pico_app/tAmeR_192.168.3.6_video_v2.apk
```

## 安装

PICO 开启开发者模式并通过 ADB 连接后，在仓库根目录执行：

```bash
adb devices
adb install -r pico_app/tAmeR_192.168.3.6_video_v2.apk
```

如果 PICO 中已经存在上游原版或其他签名版本，会出现签名不兼容错误。确认可以清除旧应用数据后，卸载再安装：

```bash
adb uninstall com.TAMEn.tAmeR
adb install pico_app/tAmeR_192.168.3.6_video_v2.apk
```

安装后在 APP 中配置控制电脑的 Wi-Fi IP 和 TCP 端口 `8018`。摄像头服务的启动方式见仓库根目录 README。
