import socket
from aura import SignalingServer, VideoStreamer
import sys
import socket
import signal
import time
from camera import ProcessingPipeline, FaceNotFoundException
import os
import numpy as np
import cv2
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

def capture_images(server, output_dir="../logs", verbose=2):
    """Capture, process, and save images with face detection"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    
    pipeline = ProcessingPipeline(
        log_path=output_dir,
        verbose=verbose
    )
    
    server.capture()
    time.sleep(2)
    
    image_bytes = server.get_capture()
    if image_bytes:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        try:        
            annotated_image = pipeline.annotate_face(image)
            processed_face = pipeline.process_image(image)
            
            # TODO: Send processed face to the model 
            
        except FaceNotFoundException:
            print(f"No face detected in captured image.")
        except Exception as e:
            print(f"Error processing image: {str(e)}")
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
            time.sleep(10)
            capture_images(server)
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down signaling server...")

if __name__ == "__main__":
    help(SignalingServer)
    help(VideoStreamer)
    