"""Run the Aura signaling server and drive emotion -> art updates."""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from aura import SignalingServer
from aura.camera import FaceNotFoundException, ProcessingPipeline
from aura.config import CONFIG


def _broadcast_emotion(server: SignalingServer, emotion: str, error: str | None = None) -> None:
    import json

    payload = {"type": "emotion", "emotion": emotion}
    if error:
        payload["error"] = error
    try:
        server.broadcast_message(json.dumps(payload))
    except Exception as exc:  # noqa: BLE001
        print(f"Emotion broadcast skipped: {exc}")


def _process_image_bytes(
    server: SignalingServer,
    image_bytes: bytes,
    output_dir: str,
    verbose: int,
) -> None:
    """Classify one captured PNG/JPEG and publish matching art."""
    os.makedirs(output_dir, exist_ok=True)
    pipeline = ProcessingPipeline(log_path=output_dir, verbose=verbose)

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print("Failed to decode captured image")
        _broadcast_emotion(server, "unknown", error="decode_failed")
        return

    try:
        pipeline.annotate_face(image)
        processed_face = pipeline.process_image(image)
    except FaceNotFoundException:
        print("No face detected in captured image.")
        _broadcast_emotion(server, "none", error="no_face")
        return
    except Exception as exc:  # noqa: BLE001 - keep server loop alive
        print(f"Error processing image: {exc}")
        _broadcast_emotion(server, "unknown", error=str(exc))
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"face_{stamp}.npy")
    np.save(out_path, processed_face)
    print(f"Saved processed face tensor to {out_path}")

    from aura.inference.runtime import EmotionArtRuntime

    runtime = EmotionArtRuntime.get_shared()
    runtime.update_from_face(processed_face)
    _broadcast_emotion(server, runtime.last_emotion or "unknown")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aura WebRTC signaling server")
    parser.add_argument(
        "--bind-ip",
        default=CONFIG.bind_ip,
        help="Interface to bind (127.0.0.1 when using the WSS proxy)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=CONFIG.signaling_port,
        help="Plain WS signaling port (streamer connects here)",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=10.0,
        help="Seconds between automatic face-capture triggers",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="How often to check for a newly arrived capture",
    )
    parser.add_argument(
        "--log-dir",
        default="../logs",
        help="Directory for face-processing logs",
    )
    args = parser.parse_args(argv)

    def _shutdown(_sig, _frame) -> None:
        print("\nShutting down signaling server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    server = SignalingServer(port=args.port, ip=args.bind_ip)
    server.start()
    print(f"Signaling server started on ws://{args.bind_ip}:{args.port}/signaling")
    print(
        "For HTTPS LAN clients, also run: "
        "python -m aura.webrtc.wss_proxy --port 3031"
    )
    print("Press Ctrl+C to stop the server")

    try:
        from aura.inference.art_generator import EmotionArtGenerator

        EmotionArtGenerator().publish("neutral", force=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not seed initial IVF art: {exc}")

    last_hash: str | None = None
    last_trigger = 0.0

    try:
        while True:
            now = time.time()
            # Ask browsers for a fresh frame on an interval.
            if now - last_trigger >= args.capture_interval:
                server.capture()
                last_trigger = now

            image_bytes = server.get_capture()
            if image_bytes:
                digest = hashlib.sha1(image_bytes).hexdigest()
                if digest != last_hash:
                    last_hash = digest
                    print(f"New capture ({len(image_bytes)} bytes) — classifying")
                    _broadcast_emotion(server, "classifying")
                    _process_image_bytes(server, image_bytes, args.log_dir, verbose=2)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nShutting down signaling server...")


if __name__ == "__main__":
    main()
