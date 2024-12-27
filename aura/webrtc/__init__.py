from aura import SignalingServer as RustSignalingServer

class SignalingServer(RustSignalingServer):
        
    """
    SignalingServer Class 
    ===============================

    The `SignalingServer` class serves as the main interface for managing WebSocket-based signaling 
    and communication between clients. This class is exported from Rust using PyO3, allowing it to 
    be used as a Python extension module.

    Attributes
    ----------
    peers : Peers
        A thread-safe, shared mapping of client IDs to their WebSocket connections.
    port : int
        The port on which the signaling server listens for incoming WebSocket connections.
    ip : str
        The IP address on which the signaling server listens for incoming WebSocket connections.
    last_captured_image : Arc<Mutex<Option[Vec[u8]>>>
        Stores the most recent image data captured from a client, encoded as raw bytes.

    Methods
    -------
    __init__(port: Optional[int] = None, ip: Optional[str] = None) -> None
        Initializes a new instance of the SignalingServer. Default port is 3030, and default IP is '127.0.0.1'.

    start() -> None
        Starts the signaling server on the specified IP and port, setting up the WebSocket route for communication.

    capture(client_id: Optional[str] = None) -> None
        Sends a signal to capture an image from a specified client or all connected clients if no client ID is provided.

    send_to_client(client_id: str, message: str) -> bool
        Sends a text message to a specific client. Returns True if successful, False otherwise.

    is_client_connected(client_id: str) -> bool
        Checks if a specific client is currently connected to the server.

    get_connected_clients() -> List[str]
        Returns a list of all currently connected client IDs.

    disconnect_client(client_id: str) -> bool
        Disconnects a specific client from the server. Returns True if successful, False otherwise.

    is_at_capacity() -> bool
        Checks if the server has reached a predefined capacity limit for connected clients.

    get_server_status() -> Dict[str, str]
        Retrieves the current server status, including IP, port, and the number of connected clients.

    broadcast_message(message: str) -> None
        Sends a text message to all connected clients.

    get_client_count() -> int
        Returns the current number of connected clients.

    get_capture() -> Optional[bytes]
        Retrieves the most recent image captured from a client, or None if no image has been captured.

    Internal Helper Functions
    -------------------------
    with_peers(peers: Peers) -> Filter
        A Warp filter for injecting the shared peers map into WebSocket routes.

    handle_connection(ws: WebSocket, peers: Peers, last_captured_image: Arc<Mutex<Option[Vec[u8]>>>) -> None
        Handles a new WebSocket connection, including message reception and peer registration.

    handle_image_message(data: str) -> None
        Processes and optionally saves image data received from a client.

    forward_message(sender_id: str, message: SignalingMessage, peers: Peers) -> None
        Forwards a signaling message to all connected clients except the sender.

    Usage
    -----
    The `SignalingServer` class provides an API for managing client connections, exchanging messages, 
    and facilitating image capture in a WebRTC signaling environment. It is designed to run a server 
    that can handle multiple WebSocket connections simultaneously.

    Example
    -------
    >>> from signaling_server import SignalingServer
    >>> server = SignalingServer(port=8080, ip="0.0.0.0")
    >>> server.start()
    >>> connected_clients = server.get_connected_clients()
    >>> print(f"Connected clients: {connected_clients}")
    """

from aura import VideoStreamer as RustVideoStreamer

class VideoStreamer(RustVideoStreamer):
    """
    The `VideoStreamer` class handles the setup and management of a WebRTC peer-to-peer connection for video streaming. 
    It allows for the streaming of video from a local source (such as an IVF file) to a WebRTC peer connection 
    and supports various WebRTC and video streaming features such as capturing screenshots, querying connection states, 
    and interacting with signaling messages.

    Attributes:
        ws_ip (str): The IP address of the WebSocket signaling server.
        ws_port (int): The port number of the WebSocket signaling server.
        ivf_dir (str): The directory where IVF video files are stored for streaming.
        peer_connection (Arc<Mutex<Option<Arc<RTCPeerConnection>>>>): A thread-safe reference to the WebRTC peer connection, wrapped in a Mutex for asynchronous access.

    Methods:
        __init__(ws_ip: str, ws_port: int, ivf_dir: str) -> VideoStreamer:
            Initializes a new `VideoStreamer` instance with the provided WebSocket IP, port, and IVF directory.

        start_streaming() -> None:
            Starts the video streaming process by establishing a WebRTC peer connection and connecting to the WebSocket signaling server.

        take_screenshot() -> bytes:
            Captures a screenshot from the currently active video stream (if a peer connection exists) and returns it as a byte array.

        get_connection_state() -> str:
            Returns the current state of the WebRTC peer connection as a string. If no active connection exists, returns an error.

        get_signaling_state() -> str:
            Returns the current signaling state of the WebRTC peer connection as a string. If no active connection exists, returns an error.

        get_stats() -> str:
            Retrieves and returns the statistics for the WebRTC peer connection as a JSON string. If no active connection exists, returns an error.

        close_connection() -> None:
            Closes the current WebRTC peer connection and resets the state. If no active connection exists, returns an error.
    """

__version__ = "0.1.0"
__all__ = ["SignalingServer", "VideoStreamer"]
