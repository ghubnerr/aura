import asyncio
import websockets
import base64

import cv2


async def main():
    # You might need to change this to 0 or a different number to pick the right camera
    cap = cv2.VideoCapture(1) 
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to capture frame from camera.")
        return

    _, buffer = cv2.imencode('.jpg', frame)
    jpg_bytes = buffer.tobytes()
    encoded_image = base64.b64encode(jpg_bytes).decode('utf-8')

    uri = "ws://localhost:6000"
    async with websockets.connect(uri) as ws:
        await ws.send(encoded_image)
        
        video_data = await ws.recv()
        with open("received_video.mp4", "wb") as f:
            f.write(video_data)

    cap = cv2.VideoCapture('received_video.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Generated Video", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

asyncio.run(main())
