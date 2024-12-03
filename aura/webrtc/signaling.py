import socket
from aura import SignalingServer
import sys
import signal
import time
import os
from datetime import datetime

def get_free_port():
    """Get an unused TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutting down signaling server...")
    sys.exit(0)

def capture_images(server, output_dir="captured_images"):
    """Trigger image capture for all peers and save images"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    server.trigger_capture()
    
    time.sleep(2)
    
    image_bytes = server.get_capture()
    if image_bytes:
        filename = os.path.join(output_dir, f"capture_{timestamp}.png")
        with open(filename, "wb") as f:
            f.write(image_bytes)
        print(f"Image saved to {filename}")
    else:
        print("No image captured")

def main():
    signal.signal(signal.SIGINT, signal_handler)

    port = 8765
    
    server = SignalingServer(port=port)
    server.start()
    
    print(f"Signaling server started on port {port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        while True:
            capture_images(server)
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down signaling server...")

if __name__ == "__main__":
    main()