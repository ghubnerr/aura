#!/usr/bin/env python3

import socket
from aura import SignalingServer
import sys
import signal
import time

def get_free_port():
    """Get an unused TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutting down signaling server...")
    sys.exit(0)

def main():
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # port = get_free_port()
    port = 8765
    
    # Create and start the signaling server
    server = SignalingServer(port=port)
    server.start()
    
    print(f"Signaling server started on port {port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down signaling server...")

if __name__ == "__main__":
    main()