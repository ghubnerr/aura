import asyncio
import websockets
import os
import cv2
import numpy as np
import torch

from .embed.model import EmotionModel  # Adjusted import
from .dataset.t2v_model import LatteT2VideoPipeline, create_prompt

# Path to your pre-defined video
PREDEFINED_VIDEO_PATH = "/disk/onyx-scratch/dullo009-fall2024/backdrops/refik/refik_sadness_532.mp4"

# Initialize Emotion Model
emotion_model = EmotionModel(pretrained=False)
model_dir = os.path.join(os.environ.get("STORAGE_PATH"), "aura_storage", "aura_emotion_classifier.pth")
emotion_model.load(model_dir)
emotion_model.eval()

# Emotion Label Mapping
label_map = {
    0: "happy",
    1: "sad",
    2: "disgust",
    3: "fear",
    4: "anger",
    5: "neutral",
    6: "happy",
    7: "neutral"
}

# Dictionary to keep track of active streaming tasks per client
active_streams = {}

async def stream_video(websocket):
    """
    Streams the pre-defined video frame by frame to the client.
    """
    cap = cv2.VideoCapture(PREDEFINED_VIDEO_PATH)
    if not cap.isOpened():
        error_message = f"ERROR: Cannot open video file at {PREDEFINED_VIDEO_PATH}."
        print(error_message)
        await websocket.send(error_message.encode('utf-8'))
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached. Looping video.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # Adjust quality as needed
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = buffer.tobytes()

            await websocket.send(frame_bytes)

            await asyncio.sleep(1 / 30)

    except websockets.ConnectionClosed:
        print("Connection closed while streaming video.")
    finally:
        cap.release()
        if websocket in active_streams:
            del active_streams[websocket]

async def handle_client(websocket):
    """
    Handles incoming client connections. For each received frame, starts streaming video if not already streaming.
    """
    print(f"Client connected from {websocket.remote_address}")

    try:
        async for message in websocket:
            # Check if the message is a binary frame
            if isinstance(message, bytes):
                print("Received a frame from client.")

                # If not already streaming, start streaming
                if websocket not in active_streams:
                    print("Starting video stream to client.")
                    stream_task = asyncio.create_task(stream_video(websocket))
                    active_streams[websocket] = stream_task
                else:
                    print("Video stream already active for this client.")

                # Optionally, process the received frame (e.g., emotion analysis)
                # This part can be customized based on your requirements
                print("Procssing input frame")
                nparr = np.frombuffer(message, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                with torch.no_grad():
                    logits = emotion_model.embed(img)
                    pred = torch.argmax(logits, dim=1).item()
                    emotion = label_map.get(pred, "neutral")
                    print(f"Predicting emotion...{emotion}")

                prompt = create_prompt(emotion)

                # If you want to send a response based on the emotion, implement it here
                # Do emotion classification and set `PREDIFINED_VIDEO_PATH`

            else:
                print(f"Received a text message: {message}")

    except websockets.ConnectionClosed:
        print(f"Client {websocket.remote_address} disconnected.")

    finally:
        if websocket in active_streams:
            active_streams[websocket].cancel()
            del active_streams[websocket]

async def main():
    async with websockets.serve(
        handle_client,
        "0.0.0.0",  # Listen on all interfaces
        6000,       # Port number
        max_size=None  # Remove size limit for incoming messages
    ):
        print("WebSocket server started on ws://localhost:6000")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
