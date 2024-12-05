import asyncio
import websockets
import base64
import os

import torch
import cv2
import numpy as np

from PIL import Image
from .embed.model import EmotionModel
from .dataset.t2v_model import LatteT2VideoPipeline, create_prompt


emotion_model = EmotionModel(pretrained=False)
emotion_model.load("output/aura_emotion_classifier.pth")
emotion_model.eval()

label_map = {0: "happy", 1: "sad", 2: "disgust", 3: "fear", 4: "anger", 5: "neutral", 6: "happy", 7: "neutral"}

async def handle_client(websocket):
    encoded_image = await websocket.recv()
    jpg_bytes = base64.b64decode(encoded_image)
    
    nparr = np.frombuffer(jpg_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[:, :, ::-1]
    
    with torch.no_grad():
        logits = emotion_model.embed(img)
        pred = torch.argmax(logits, dim=1).item()
        emotion = label_map[pred]

    prompt = create_prompt(emotion)
    
    t2v_pipeline = LatteT2VideoPipeline(enable_pab=True)
    output_path = "temp_output.mp4"
    t2v_pipeline(prompt, path=output_path, steps=30)

    with open(output_path, "rb") as f:
        video_bytes = f.read()
    await websocket.send(video_bytes)

    os.remove(output_path)

async def main():
    async with websockets.serve(handle_client, "", 6000):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
