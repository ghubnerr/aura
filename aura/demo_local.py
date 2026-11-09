"""Send one webcam frame to the legacy demo websocket server."""

from __future__ import annotations

import argparse
import asyncio

import cv2
import websockets


async def run(camera_index: int, uri: str) -> None:
    capture = cv2.VideoCapture(camera_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit("Failed to capture frame from camera.")

    _ok, buffer = cv2.imencode(".jpg", frame)
    async with websockets.connect(uri, max_size=None) as websocket:
        await websocket.send(buffer.tobytes())
        print("Frame sent; waiting for streamed JPEG frames (Ctrl+C to stop)...")
        async for message in websocket:
            if isinstance(message, (bytes, bytearray)):
                print(f"Received frame bytes: {len(message)}")
            else:
                print(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--uri", default="ws://127.0.0.1:6000")
    args = parser.parse_args()
    asyncio.run(run(args.camera, args.uri))


if __name__ == "__main__":
    main()
