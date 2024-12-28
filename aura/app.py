from flask import Flask, request, jsonify
from flask_cors import CORS
from aura import SignalingServer, VideoStreamer
import os
import threading
from werkzeug.serving import make_server

WEBSOCKET_PORT = 8765
FLASK_PORT = 5000

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
         "origins": [
            "https://aura-8nhxn8a2f-ghubnerrs-projects.vercel.app",
            "http://localhost:3000"
        ]
    }
})

signaling_server = SignalingServer(port=WEBSOCKET_PORT)
video_streamer = VideoStreamer(
    ws_ip="0.0.0.0",  
    ws_port=WEBSOCKET_PORT,
    ivf_dir="./output"
)

def run_signaling():
    try:
        signaling_server.start()
    except Exception as e:
        print(f"Signaling server error: {e}")

def run_streaming():
    try:
        video_streamer.start_streaming()
    except Exception as e:
        print(f"Video streamer error: {e}")

def run_flask():
    try:
        server = make_server('0.0.0.0', FLASK_PORT, app)
        server.serve_forever()
    except Exception as e:
        print(f"Flask server error: {e}")

if __name__ == '__main__':
    # Start signaling server in its own thread
    signaling_thread = threading.Thread(target=run_signaling)
    signaling_thread.daemon = True
    signaling_thread.start()

    # Start video streamer in its own thread
    streaming_thread = threading.Thread(target=run_streaming)
    streaming_thread.daemon = True
    streaming_thread.start()

    # Start Flask server in the main thread
    run_flask()
