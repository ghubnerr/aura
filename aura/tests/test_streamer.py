import os
import socket
import shutil
import pytest
import asyncio
import websockets
import time
import threading
from contextlib import closing
from aura.webrtc import (
    VideoStreamer,
    DEFAULT_WS_IP,
    DEFAULT_TEST_IVF_DIR,
    TEST_IP_ADDRESSES,
)


@pytest.fixture(autouse=True)
def setup_video_dir():
    video_dir = DEFAULT_TEST_IVF_DIR
    os.makedirs(video_dir, exist_ok=True)
    yield
    shutil.rmtree(video_dir, ignore_errors=True)


@pytest.fixture
def free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def streamer(free_port):
    streamer = VideoStreamer(
        ws_ip=DEFAULT_WS_IP, ws_port=free_port, ivf_dir=DEFAULT_TEST_IVF_DIR
    )
    yield streamer
    try:
        streamer.close_connection()
    except RuntimeError:
        pass


@pytest.fixture
async def mock_websocket_server(free_port):
    async def handler(websocket):
        async for message in websocket:
            await websocket.send(message)

    server = await websockets.serve(
        handler, DEFAULT_WS_IP, free_port, reuse_address=True, reuse_port=True
    )
    yield server
    server.close()
    await server.wait_closed()


@pytest.fixture
def sample_ivf_file(tmp_path):
    ivf_path = tmp_path / "test.ivf"
    with open(ivf_path, "wb") as f:
        f.write(b"DKIF\x00\x00\x00\x00")
        f.write(b"\x00" * 24)
    return ivf_path


@pytest.fixture
def valid_ivf_file(tmp_path):
    ivf_path = tmp_path / "valid.ivf"
    with open(ivf_path, "wb") as f:
        f.write(b"DKIF")
        f.write((0).to_bytes(2, byteorder="little"))
        f.write((32).to_bytes(2, byteorder="little"))
        f.write(b"VP80")
        f.write((640).to_bytes(2, byteorder="little"))
        f.write((480).to_bytes(2, byteorder="little"))
        f.write((1).to_bytes(4, byteorder="little"))
        f.write((30).to_bytes(4, byteorder="little"))
        f.write((100).to_bytes(4, byteorder="little"))
        f.write((0).to_bytes(4, byteorder="little"))
    return ivf_path


@pytest.fixture
def mock_signaling_server(free_port):
    received_messages = []

    async def handler(websocket, path):
        try:
            async for message in websocket:
                received_messages.append(message)
                await websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            pass

    server = None

    async def start_server():
        nonlocal server
        server = await websockets.serve(
            handler, DEFAULT_WS_IP, free_port, reuse_address=True
        )
        return server

    def get_messages():
        return received_messages.copy()

    return start_server, get_messages


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


class TestVideoStreamerInitialization:
    def test_streamer_initialization(self, free_port):
        streamer = VideoStreamer(
            ws_ip=DEFAULT_WS_IP, ws_port=free_port, ivf_dir=DEFAULT_TEST_IVF_DIR
        )
        assert streamer is not None

    def test_streamer_initialization_with_different_ips(self, free_port):
        for ip in TEST_IP_ADDRESSES:
            streamer = VideoStreamer(
                ws_ip=ip, ws_port=free_port, ivf_dir=DEFAULT_TEST_IVF_DIR
            )
            assert streamer is not None

    def test_streamer_initialization_with_different_ports(self):
        for port in [8080, 9000, 3000]:
            streamer = VideoStreamer(
                ws_ip=DEFAULT_WS_IP, ws_port=port, ivf_dir=DEFAULT_TEST_IVF_DIR
            )
            assert streamer is not None

    def test_streamer_initialization_with_different_directories(
        self, free_port, tmp_path
    ):
        test_dirs = [DEFAULT_TEST_IVF_DIR, str(tmp_path / "custom"), "/tmp/test_stream"]
        for directory in test_dirs:
            os.makedirs(directory, exist_ok=True)
            streamer = VideoStreamer(
                ws_ip=DEFAULT_WS_IP, ws_port=free_port, ivf_dir=directory
            )
            assert streamer is not None


class TestVideoStreamerErrorHandling:
    def test_get_stats_no_connection_detailed(self, streamer):
        with pytest.raises(RuntimeError) as exc_info:
            streamer.get_stats()
        assert "No active peer connection" in str(exc_info.value)

    def test_close_connection_no_connection_detailed(self, streamer):
        with pytest.raises(RuntimeError) as exc_info:
            streamer.close_connection()
        assert "No active peer connection" in str(exc_info.value)

    def test_multiple_close_connection_calls(self, streamer):
        with pytest.raises(RuntimeError, match="No active peer connection"):
            streamer.close_connection()

        with pytest.raises(RuntimeError, match="No active peer connection"):
            streamer.close_connection()

    def test_get_stats_after_close_connection(self, streamer):
        with pytest.raises(RuntimeError, match="No active peer connection"):
            streamer.close_connection()

        with pytest.raises(RuntimeError, match="No active peer connection"):
            streamer.get_stats()


class TestVideoStreamerStreaming:
    def test_start_streaming_does_not_raise(self, streamer):
        streamer.start_streaming()

    def test_start_streaming_multiple_times(self, streamer):
        streamer.start_streaming()
        streamer.start_streaming()

    def test_start_streaming_with_invalid_directory(self, free_port, tmp_path):
        invalid_dir = str(tmp_path / "nonexistent")
        streamer = VideoStreamer(
            ws_ip=DEFAULT_WS_IP, ws_port=free_port, ivf_dir=invalid_dir
        )

        streamer.start_streaming()


class TestFileMonitoring:
    def test_video_directory_monitoring_with_multiple_files(
        self, streamer, sample_ivf_file
    ):
        streamer.start_streaming()

        for i in range(3):
            shutil.copy(sample_ivf_file, f"/tmp/video/test_{i}.ivf")
            time.sleep(0.1)

        time.sleep(1)

        for i in range(3):
            os.remove(f"/tmp/video/test_{i}.ivf")

    def test_video_directory_monitoring_with_non_ivf_files(self, streamer, tmp_path):
        streamer.start_streaming()

        non_ivf_files = ["test.txt", "test.mp4", "test.avi", "test.mov"]
        for filename in non_ivf_files:
            with open(f"/tmp/video/{filename}", "w") as f:
                f.write("test content")
            time.sleep(0.1)

        time.sleep(1)

        for filename in non_ivf_files:
            os.remove(f"/tmp/video/{filename}")

    def test_video_directory_monitoring_with_valid_ivf(self, streamer, valid_ivf_file):
        streamer.start_streaming()
        shutil.copy(valid_ivf_file, "/tmp/video/valid_test.ivf")
        time.sleep(1)
        os.remove("/tmp/video/valid_test.ivf")

    def test_video_directory_monitoring_with_corrupted_ivf(self, streamer, tmp_path):
        streamer.start_streaming()

        corrupted_ivf = tmp_path / "corrupted.ivf"
        with open(corrupted_ivf, "wb") as f:
            f.write(b"INVALID_HEADER")

        shutil.copy(corrupted_ivf, "/tmp/video/corrupted.ivf")
        time.sleep(1)
        os.remove("/tmp/video/corrupted.ivf")

    def test_video_directory_monitoring_file_removal(self, streamer, sample_ivf_file):
        streamer.start_streaming()

        shutil.copy(sample_ivf_file, "/tmp/video/test.ivf")
        time.sleep(0.5)

        os.remove("/tmp/video/test.ivf")
        time.sleep(0.5)

        shutil.copy(sample_ivf_file, "/tmp/video/test.ivf")
        time.sleep(0.5)
        os.remove("/tmp/video/test.ivf")


class TestWebSocketIntegration:
    @pytest.mark.asyncio
    async def test_streamer_with_mock_websocket_server(
        self, free_port, sample_ivf_file
    ):
        async def handler(websocket, path):
            async for message in websocket:
                await websocket.send(message)

        server = await websockets.serve(
            handler, DEFAULT_WS_IP, free_port, reuse_address=True
        )

        try:
            streamer = VideoStreamer(
                ws_ip=DEFAULT_WS_IP, ws_port=free_port, ivf_dir=DEFAULT_TEST_IVF_DIR
            )
            streamer.start_streaming()

            await asyncio.sleep(1)

            shutil.copy(sample_ivf_file, "/tmp/video/test.ivf")
            await asyncio.sleep(1)

            os.remove("/tmp/video/test.ivf")

        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_streamer_connection_failure(self, free_port, sample_ivf_file):
        streamer = VideoStreamer(
            ws_ip=DEFAULT_WS_IP, ws_port=free_port, ivf_dir=DEFAULT_TEST_IVF_DIR
        )
        streamer.start_streaming()

        await asyncio.sleep(1)

        shutil.copy(sample_ivf_file, "/tmp/video/test.ivf")
        await asyncio.sleep(1)
        os.remove("/tmp/video/test.ivf")


class TestConcurrentOperations:
    def test_concurrent_file_operations(self, streamer, sample_ivf_file):
        streamer.start_streaming()

        def add_file(file_num):
            shutil.copy(sample_ivf_file, f"/tmp/video/concurrent_{file_num}.ivf")
            time.sleep(0.1)
            os.remove(f"/tmp/video/concurrent_{file_num}.ivf")

        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_file, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        time.sleep(1)

    def test_start_streaming_while_processing_files(self, streamer, sample_ivf_file):
        streamer.start_streaming()

        shutil.copy(sample_ivf_file, "/tmp/video/test.ivf")
        time.sleep(0.5)

        streamer.start_streaming()

        shutil.copy(sample_ivf_file, "/tmp/video/test2.ivf")
        time.sleep(0.5)

        os.remove("/tmp/video/test.ivf")
        os.remove("/tmp/video/test2.ivf")


class TestEdgeCases:
    def test_empty_directory_monitoring(self, streamer):
        streamer.start_streaming()
        time.sleep(1)

    def test_very_large_ivf_file(self, streamer, tmp_path):
        large_ivf = tmp_path / "large.ivf"
        with open(large_ivf, "wb") as f:
            f.write(b"DKIF\x00\x00\x00\x00")
            f.write(b"\x00" * 24)
            f.write(b"\x00" * 1024 * 1024)

        streamer.start_streaming()
        shutil.copy(large_ivf, "/tmp/video/large.ivf")
        time.sleep(1)
        os.remove("/tmp/video/large.ivf")

    def test_rapid_file_changes(self, streamer, sample_ivf_file):
        streamer.start_streaming()

        for i in range(10):
            shutil.copy(sample_ivf_file, f"/tmp/video/rapid_{i}.ivf")
            time.sleep(0.05)
            os.remove(f"/tmp/video/rapid_{i}.ivf")
            time.sleep(0.05)

    def test_streamer_with_special_characters_in_path(self, free_port, tmp_path):
        special_dir = str(tmp_path / "test with spaces & symbols!")
        os.makedirs(special_dir, exist_ok=True)

        streamer = VideoStreamer(
            ws_ip=DEFAULT_WS_IP, ws_port=free_port, ivf_dir=special_dir
        )
        streamer.start_streaming()
        time.sleep(1)


class TestCleanupAndTeardown:
    def test_streamer_cleanup_in_fixture(self, free_port):
        streamer = VideoStreamer(
            ws_ip=DEFAULT_WS_IP, ws_port=free_port, ivf_dir=DEFAULT_TEST_IVF_DIR
        )
        streamer.start_streaming()

    def test_multiple_streamers_cleanup(self, free_port):
        streamers = []
        for i in range(3):
            streamer = VideoStreamer(
                ws_ip=DEFAULT_WS_IP, ws_port=free_port + i, ivf_dir=f"/tmp/video_{i}"
            )
            streamer.start_streaming()
            streamers.append(streamer)

        for streamer in streamers:
            try:
                streamer.close_connection()
            except RuntimeError:
                pass
