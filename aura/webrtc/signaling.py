"""Run the Aura signaling server and drive emotion → art updates."""

from __future__ import annotations

import argparse
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


def _process_capture(server: SignalingServer, output_dir: str, verbose: int) -> None:
    """Ask clients for a frame, crop the face, and publish emotion art."""
    os.makedirs(output_dir, exist_ok=True)
    pipeline = ProcessingPipeline(log_path=output_dir, verbose=verbose)

    server.capture()
    time.sleep(2)

    image_bytes = server.get_capture()
    if not image_bytes:
        print("No image captured")
        return

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print("Failed to decode captured image")
        return

    try:
        pipeline.annotate_face(image)
        processed_face = pipeline.process_image(image)
        _handle_processed_face(processed_face, output_dir)
    except FaceNotFoundException:
        print("No face detected in captured image.")
    except Exception as exc:  # noqa: BLE001 - keep server loop alive
        print(f"Error processing image: {exc}")


def _handle_processed_face(processed_face: np.ndarray, output_dir: str) -> None:
    """Persist the face crop and update the live emotion-conditioned art."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"face_{stamp}.npy")
    np.save(out_path, processed_face)
    print(f"Saved processed face tensor to {out_path}")

    from aura.inference.runtime import EmotionArtRuntime

    EmotionArtRuntime.get_shared().update_from_face(processed_face)


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
        help="Seconds between automatic face captures",
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

    try:
        while True:
            time.sleep(args.capture_interval)
            _process_capture(server, args.log_dir, verbose=2)
    except KeyboardInterrupt:
        print("\nShutting down signaling server...")


if __name__ == "__main__":
    main()
