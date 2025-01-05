from flask import Flask, request, jsonify
from flask_cors import CORS
from aura import SignalingServer, VideoStreamer
import os
import threading
from werkzeug.serving import make_server
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/a/buffalo.cs.fiu.edu./disk/jccl-002/homes/glucc002/nginx/logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

WEBSOCKET_PORT = 8765
FLASK_PORT = 5000

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
         "origins": [
            "https://aura-8nhxn8a2f-ghubnerrs-projects.vercel.app",
            "https://aura-ism2p97vc-ghubnerrs-projects.vercel.app"
            "http://localhost:3000"
        ]
    }
})

signaling_server = SignalingServer(port=WEBSOCKET_PORT, ip="0.0.0.0")
video_streamer = VideoStreamer(
    ws_ip="0.0.0.0",  
    ws_port=WEBSOCKET_PORT,
    ivf_dir="./output"
)
def run_signaling():
    try:
        logger.info("Starting signaling server on port %d", WEBSOCKET_PORT)
        signaling_server.start()
    except Exception as e:
        logger.error("Signaling server error: %s", str(e), exc_info=True)

def run_streaming():
    try:
        logger.info("Starting video streamer")
        video_streamer.start_streaming()
    except Exception as e:
        logger.error("Video streamer error: %s", str(e), exc_info=True)

def run_flask():
    try:
        logger.info("Starting Flask server on port %d", FLASK_PORT)
        server = make_server('0.0.0.0', FLASK_PORT, app)
        server.serve_forever()
    except Exception as e:
        logger.error("Flask server error: %s", str(e), exc_info=True)
        
if __name__ == '__main__':
    # Start signaling server in its own thread
    signaling_thread = threading.Thread(target=run_signaling)
    signaling_thread.daemon = False  # Set daemon status before starting
    signaling_thread.start()

    # Start video streamer in its own thread
    streaming_thread = threading.Thread(target=run_streaming)
    streaming_thread.daemon = False  # Set daemon status before starting
    streaming_thread.start()

    # Start Flask server in the main thread
    run_flask()
