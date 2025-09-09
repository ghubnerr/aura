from aura import SignalingServer
from aura import VideoStreamer

__version__ = "0.1.0"

# WebSocket and server configuration
DEFAULT_WS_IP = "127.0.0.1"
DEFAULT_WS_PORT = 3030
DEFAULT_SIGNALING_PORT = 3030

# Directory paths
DEFAULT_IVF_DIR = "./ivf_files"
DEFAULT_LOGS_DIR = "../logs"
DEFAULT_TEST_IVF_DIR = "/tmp/video"

# Processing configuration
DEFAULT_VERBOSE_LEVEL = 2

# Timing configuration
CAPTURE_DELAY_SECONDS = 2
SERVER_SLEEP_SECONDS = 10

# Test configuration
TEST_IP_ADDRESSES = ["127.0.0.1", "localhost", "0.0.0.0"]

__all__ = [
    "SignalingServer",
    "VideoStreamer",
    "DEFAULT_WS_IP",
    "DEFAULT_WS_PORT",
    "DEFAULT_SIGNALING_PORT",
    "DEFAULT_IVF_DIR",
    "DEFAULT_LOGS_DIR",
    "DEFAULT_TEST_IVF_DIR",
    "DEFAULT_VERBOSE_LEVEL",
    "CAPTURE_DELAY_SECONDS",
    "SERVER_SLEEP_SECONDS",
    "TEST_IP_ADDRESSES",
]
