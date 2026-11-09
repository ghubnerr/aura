"""Run the Aura signaling server and drive emotion -> art updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from aura import SignalingServer
from aura.camera import FaceNotFoundException, ProcessingPipeline
from aura.config import CONFIG


def _broadcast(server: SignalingServer, payload: dict) -> None:
    try:
        server.broadcast_message(json.dumps(payload))
    except Exception as exc:  # noqa: BLE001
        print(f"Broadcast skipped: {exc}")


def _broadcast_emotion(
    server: SignalingServer,
    emotion: str,
    error: str | None = None,
    demo: bool = False,
) -> None:
    payload: dict = {"type": "emotion", "emotion": emotion, "demo": demo}
    if error:
        payload["error"] = error
    _broadcast(server, payload)


def _process_image_bytes(
    server: SignalingServer,
    image_bytes: bytes,
    output_dir: str,
    verbose: int,
) -> None:
    """Classify one captured PNG/JPEG and condition the continuous art engine."""
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
    _broadcast_emotion(
        server,
        runtime.last_emotion or "unknown",
        demo=runtime.use_demo_heuristic,
    )


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
        default=8.0,
        help="Seconds between automatic face-capture triggers",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
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

    # Continuous path starts in the background (never blocks capture loop).
    try:
        from aura.inference.continuous_art import ContinuousArtEngine

        engine = ContinuousArtEngine.get_shared()
        engine.start()
        _broadcast(
            server,
            {
                "type": "art_status",
                "backend": type(engine.backend).__name__,
                "fps": engine.fps,
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Continuous art engine failed to start: {exc}")
        try:
            from aura.inference.continuous_art import seed_live_ivf

            seed_live_ivf()
        except Exception as seed_exc:  # noqa: BLE001
            print(f"Could not seed initial IVF art: {seed_exc}")

    last_hash: str | None = None
    last_trigger = 0.0
    classify_lock = threading.Lock()

    def _classify_async(image_bytes: bytes) -> None:
        if not classify_lock.acquire(blocking=False):
            print("Classification already running; skipping overlapping capture")
            return
        try:
            _process_image_bytes(server, image_bytes, args.log_dir, verbose=1)
        finally:
            classify_lock.release()

    try:
        while True:
            now = time.time()
            if now - last_trigger >= args.capture_interval:
                server.capture()
                last_trigger = now

            image_bytes = server.get_capture()
            if image_bytes:
                digest = hashlib.sha1(image_bytes).hexdigest()
                if digest != last_hash:
                    last_hash = digest
                    print(f"New capture ({len(image_bytes)} bytes) - classifying")
                    _broadcast_emotion(server, "classifying")
                    threading.Thread(
                        target=_classify_async,
                        args=(image_bytes,),
                        name="aura-classify",
                        daemon=True,
                    ).start()

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nShutting down signaling server...")


if __name__ == "__main__":
    main()
