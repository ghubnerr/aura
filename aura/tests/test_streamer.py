import os
import socket
import shutil
import pytest
import asyncio
import websockets
from contextlib import closing
from aura.webrtc import VideoStreamer

@pytest.fixture(autouse=True)
def setup_video_dir():
    video_dir = "/tmp/video"
    os.makedirs(video_dir, exist_ok=True)
    yield
    shutil.rmtree(video_dir, ignore_errors=True)

@pytest.fixture
def free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        return s.getsockname()[1]

@pytest.fixture
def streamer(free_port):
    streamer = VideoStreamer(
        ws_ip="127.0.0.1",
        ws_port=free_port,
        ivf_dir="/tmp/video"
    )
    yield streamer
    try:
        streamer.close_connection()
    except:
        pass

@pytest.fixture
async def mock_websocket_server(free_port):
    async def handler(websocket):
        async for message in websocket:
            await websocket.send(message)
    
    server = await websockets.serve(
        handler, 
        "127.0.0.1", 
        free_port,
        reuse_address=True,
        reuse_port=True
    )
    yield server
    server.close()
    await server.wait_closed()

@pytest.fixture
def sample_ivf_file(tmp_path):
    ivf_path = tmp_path / "test.ivf"
    with open(ivf_path, "wb") as f:
        f.write(b"DKIF\x00\x00\x00\x00")
        f.write(b"\x00" * 24)  # Dummy header data
    return ivf_path

@pytest.mark.asyncio
async def test_start_streaming(streamer, mock_websocket_server):
    try:
        async with asyncio.timeout(5):  
            streamer.start_streaming()
            
            # Wait for connection establishment
            for _ in range(50):  
                if streamer.get_connection_state() != "new":
                    break
                await asyncio.sleep(0.1)
            
            assert streamer.get_connection_state() in ("connecting", "connected")
            assert streamer.get_signaling_state() == "stable"
    finally:
        # Ensure cleanup
        await streamer.close_connection()

def test_take_screenshot_no_connection(streamer):
    with pytest.raises(RuntimeError, match="No active peer connection"):
        streamer.take_screenshot()

def test_get_stats_no_connection(streamer):
    with pytest.raises(RuntimeError, match="No active peer connection"):
        streamer.get_stats()

def test_close_connection_no_connection(streamer):
    with pytest.raises(RuntimeError, match="No active peer connection"):
        streamer.close_connection()

def test_video_directory_monitoring(streamer, sample_ivf_file):
    streamer.start_streaming()
    shutil.copy(sample_ivf_file, "/tmp/video/test.ivf")
    import time
    time.sleep(1)
    os.remove("/tmp/video/test.ivf")
