"""Legacy websocket demo that classifies a face and streams a canned clip.

Prefer the Rust WebRTC path (`python -m aura.webrtc.signaling` + streamer).
This script remains for quick emotion-classifier smoke tests on a GPU box.
"""

from __future__ import annotations

import asyncio
import os
from typing import Dict

import cv2
import numpy as np
import torch
import websockets

from aura.cnn import EmotionModel
from aura.config import CONFIG
from aura.dataset.t2v_model import create_prompt

PREDEFINED_VIDEO_PATH = os.environ.get(
    "AURA_DEMO_VIDEO",
    os.path.join(CONFIG.storage_path, "demo", "refik_sadness.mp4"),
)

LABEL_MAP: Dict[int, str] = {
    0: "happy",
    1: "sad",
    2: "anger",
    3: "disgust",
    4: "fear",
    5: "neutral",
    6: "surprise",
    7: "contempt",
}

emotion_model = EmotionModel(pretrained=False)
if os.path.isfile(CONFIG.emotion_checkpoint):
    emotion_model.load(CONFIG.emotion_checkpoint)
emotion_model.eval()

active_streams: Dict[websockets.WebSocketServerProtocol, asyncio.Task] = {}


async def stream_video(websocket: websockets.WebSocketServerProtocol) -> None:
    capture = cv2.VideoCapture(PREDEFINED_VIDEO_PATH)
    if not capture.isOpened():
        await websocket.send(
            f"ERROR: Cannot open video file at {PREDEFINED_VIDEO_PATH}.".encode("utf-8")
        )
        return

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            _ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            await websocket.send(buffer.tobytes())
            await asyncio.sleep(1 / 30)
    except websockets.ConnectionClosed:
        print("Connection closed while streaming video.")
    finally:
        capture.release()
        active_streams.pop(websocket, None)


async def handle_client(websocket: websockets.WebSocketServerProtocol) -> None:
    print(f"Client connected from {websocket.remote_address}")
    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                print(f"Received text message: {message}")
                continue

            if websocket not in active_streams:
                active_streams[websocket] = asyncio.create_task(stream_video(websocket))

            image = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue

            with torch.no_grad():
                logits = emotion_model.embed(image, use_representation=False)
                pred = int(torch.argmax(logits, dim=1).item())
                emotion = LABEL_MAP.get(pred, "neutral")
                print(f"Predicted emotion: {emotion}")
                print(f"Prompt: {create_prompt(emotion)}")
    except websockets.ConnectionClosed:
        print(f"Client {websocket.remote_address} disconnected.")
    finally:
        task = active_streams.pop(websocket, None)
        if task is not None:
            task.cancel()


async def main() -> None:
    async with websockets.serve(handle_client, "0.0.0.0", 6000, max_size=None):
        print("WebSocket demo server started on ws://0.0.0.0:6000")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
