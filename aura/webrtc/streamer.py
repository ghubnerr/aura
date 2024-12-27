import socket
import sys
import signal
import time
from ..webrtc import VideoStreamer

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutting down video streamer...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)

    ws_ip = "127.0.0.1"  # WebSocket IP address
    ws_port = 8765       # WebSocket port
    ivf_dir = "../ivf_dir"    # Directory to watch for IVF files

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
    
