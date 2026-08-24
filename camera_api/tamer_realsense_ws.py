#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import json
import threading
import time
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
from aiohttp import web

from D435_rgb_depth import D435Camera, list_realsense_devices


class RealSenseStripHub:
    def __init__(self, args):
        self.args = args
        self.cameras: List[D435Camera] = []
        self.threads: List[threading.Thread] = []
        self.frames: Dict[str, Optional[np.ndarray]] = {}
        self.last_update: Dict[str, Optional[float]] = {}
        self.errors: Dict[str, Optional[str]] = {}
        self.lock = threading.Lock()
        self.running = False
        self.clients: Set[web.WebSocketResponse] = set()

    def start(self) -> None:
        available = list_realsense_devices()
        available_serials = [device["serial"] for device in available]
        requested_serials = self.args.serial or available_serials[: self.args.max_cameras]

        if not requested_serials:
            raise RuntimeError("未检测到 RealSense 相机")

        unknown_serials = [serial for serial in requested_serials if serial not in available_serials]
        if unknown_serials:
            raise RuntimeError(f"未检测到指定相机: {', '.join(unknown_serials)}")

        self.running = True
        for serial in requested_serials[: self.args.max_cameras]:
            camera = D435Camera(
                serial=serial,
                width=self.args.capture_width,
                height=self.args.capture_height,
                fps=self.args.capture_fps,
                enable_color=True,
                enable_depth=False,
                align_depth_to_color=False,
            )
            try:
                camera.start()
            except Exception:
                self.stop()
                raise

            active_serial = camera.get_active_device_serial()
            self.cameras.append(camera)
            self.frames[active_serial] = None
            self.last_update[active_serial] = None
            self.errors[active_serial] = None

            thread = threading.Thread(
                target=self._capture_loop,
                args=(camera, active_serial),
                daemon=True,
                name=f"realsense-{active_serial}",
            )
            thread.start()
            self.threads.append(thread)

        print("视频相机顺序:")
        for index, camera in enumerate(self.cameras, start=1):
            print(f"  {index}: {camera.get_active_device_serial()}")

    def _capture_loop(self, camera: D435Camera, serial: str) -> None:
        while self.running:
            try:
                data = camera.get_frames(timeout_ms=2000)
                frame = data["color"]
                if frame is None:
                    continue
                with self.lock:
                    self.frames[serial] = frame
                    self.last_update[serial] = time.time()
                    self.errors[serial] = None
            except Exception as exc:
                if self.running:
                    with self.lock:
                        self.errors[serial] = str(exc)
                    time.sleep(0.1)

    def stop(self) -> None:
        self.running = False
        for camera in self.cameras:
            camera.stop()
        for thread in self.threads:
            thread.join(timeout=2.5)
        self.cameras.clear()
        self.threads.clear()

    def build_strip(self) -> np.ndarray:
        tiles = []
        with self.lock:
            frames = [self.frames.get(camera.get_active_device_serial()) for camera in self.cameras]

        for frame in frames:
            if frame is None:
                tile = np.zeros((self.args.tile_height, self.args.tile_width, 3), dtype=np.uint8)
            else:
                tile = cv2.resize(
                    frame,
                    (self.args.tile_width, self.args.tile_height),
                    interpolation=cv2.INTER_AREA,
                )
            tiles.append(tile)

        while len(tiles) < self.args.output_tiles:
            tiles.append(np.zeros((self.args.tile_height, self.args.tile_width, 3), dtype=np.uint8))

        return np.hstack(tiles[: self.args.output_tiles])

    def encode_frame(self) -> Optional[bytes]:
        strip = self.build_strip()
        ok, encoded = cv2.imencode(
            ".jpg",
            strip,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality],
        )
        return encoded.tobytes() if ok else None

    def health(self) -> Dict:
        now = time.time()
        with self.lock:
            cameras = [
                {
                    "index": index,
                    "serial": camera.get_active_device_serial(),
                    "frame_age_seconds": (
                        None
                        if self.last_update.get(camera.get_active_device_serial()) is None
                        else round(now - self.last_update[camera.get_active_device_serial()], 3)
                    ),
                    "error": self.errors.get(camera.get_active_device_serial()),
                }
                for index, camera in enumerate(self.cameras, start=1)
            ]
        return {
            "ok": bool(cameras) and all(camera["frame_age_seconds"] is not None for camera in cameras),
            "cameras": cameras,
            "clients": len(self.clients),
            "video_ws": "/ws/video",
        }

    async def broadcast_loop(self) -> None:
        frame_interval = 1.0 / max(1, self.args.stream_fps)
        while True:
            if self.clients:
                payload = self.encode_frame()
                if payload is not None:
                    dead_clients = []
                    for websocket in list(self.clients):
                        try:
                            await websocket.send_bytes(payload)
                        except Exception:
                            dead_clients.append(websocket)
                    for websocket in dead_clients:
                        with contextlib.suppress(Exception):
                            await websocket.close()
                        self.clients.discard(websocket)
            await asyncio.sleep(frame_interval)


def make_app(args):
    hub = RealSenseStripHub(args)

    async def index(_request):
        return web.json_response(
            {
                "message": "RealSense WebSocket-JPEG bridge is running",
                "health": "/health",
                "video_ws": "/ws/video",
            }
        )

    async def health(_request):
        return web.json_response(hub.health())

    async def config(_request):
        return web.json_response(
            {
                "mode": "websocket_jpeg_strip",
                "camera_count": len(hub.cameras),
                "strip_size": {
                    "width": args.tile_width * args.output_tiles,
                    "height": args.tile_height,
                },
                "tile_size": {
                    "width": args.tile_width,
                    "height": args.tile_height,
                },
                "fps": args.stream_fps,
                "jpeg_quality": args.jpeg_quality,
                "video_ws": "/ws/video",
            }
        )

    async def video_ws(request):
        websocket = web.WebSocketResponse(max_msg_size=2 * 1024 * 1024)
        await websocket.prepare(request)
        hub.clients.add(websocket)
        await websocket.send_str(
            json.dumps(
                {
                    "type": "hello",
                    "mode": "websocket_jpeg_strip",
                    "width": args.tile_width * args.output_tiles,
                    "height": args.tile_height,
                    "fps": args.stream_fps,
                }
            )
        )

        try:
            async for message in websocket:
                if message.type == web.WSMsgType.TEXT and message.data == "ping":
                    await websocket.send_str("pong")
                elif message.type == web.WSMsgType.ERROR:
                    break
        finally:
            hub.clients.discard(websocket)
            with contextlib.suppress(Exception):
                await websocket.close()
        return websocket

    async def on_startup(_app):
        hub.start()

    async def stream_context(_app):
        task = asyncio.create_task(hub.broadcast_loop())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def on_shutdown(_app):
        for websocket in list(hub.clients):
            with contextlib.suppress(Exception):
                await websocket.close()
        hub.clients.clear()
        await asyncio.to_thread(hub.stop)

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/config", config)
    app.router.add_get("/ws/video", video_ws)
    app.on_startup.append(on_startup)
    app.cleanup_ctx.append(stream_context)
    app.on_shutdown.append(on_shutdown)
    return app


def parse_args():
    parser = argparse.ArgumentParser(description="RealSense cameras to tAmeR WebSocket-JPEG bridge")
    parser.add_argument(
        "--serial",
        action="append",
        help="按显示顺序指定相机序列号，可重复传入；默认使用检测到的前三台",
    )
    parser.add_argument("--max-cameras", type=int, default=3)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--capture-fps", type=int, default=30)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=240)
    parser.add_argument("--output-tiles", type=int, default=4)
    parser.add_argument("--stream-fps", type=int, default=10)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    return parser.parse_args()


def main():
    args = parse_args()
    web.run_app(make_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
