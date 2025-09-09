import signal
import sys
import time

from aura import VideoStreamer
from aura.webrtc import DEFAULT_WS_IP, DEFAULT_WS_PORT, DEFAULT_IVF_DIR


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutting down video streamer...")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)

    ws_ip = DEFAULT_WS_IP
    ws_port = DEFAULT_WS_PORT
    ivf_dir = DEFAULT_IVF_DIR

    streamer = VideoStreamer(ws_ip, ws_port, ivf_dir)
    streamer.start_streaming()

    print(f"Video streamer started - WebSocket server at ws://{ws_ip}:{ws_port}")
    print(f"Watching directory: {ivf_dir}")
    print("Press Ctrl+C to stop the streamer")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down video streamer...")


if __name__ == "__main__":
    main()
