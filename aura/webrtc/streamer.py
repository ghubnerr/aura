"""Run the Aura IVF → WebRTC video streamer."""

from __future__ import annotations

import argparse
import signal
import sys
import time

from aura import VideoStreamer
from aura.config import CONFIG


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aura WebRTC IVF streamer")
    parser.add_argument(
        "--signaling-host",
        default=CONFIG.signaling_host,
        help="Signaling server host the streamer should join",
    )
    parser.add_argument(
        "--signaling-port",
        type=int,
        default=CONFIG.signaling_port,
        help="Signaling server port",
    )
    parser.add_argument(
        "--ivf-dir",
        default=CONFIG.ivf_dir,
        help="Directory watched for modified .ivf art clips",
    )
    args = parser.parse_args(argv)

    def _shutdown(_sig, _frame) -> None:
        print("\nShutting down video streamer...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    streamer = VideoStreamer(args.signaling_host, args.signaling_port, args.ivf_dir)
    streamer.start_streaming()

    print(
        "Video streamer started — joining "
        f"ws://{args.signaling_host}:{args.signaling_port}/signaling"
    )
    print(f"Watching directory: {args.ivf_dir}")
    print("Press Ctrl+C to stop the streamer")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down video streamer...")


if __name__ == "__main__":
    main()
