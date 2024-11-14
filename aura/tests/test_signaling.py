import pytest
import asyncio
import websockets
import json
from aura import SignalingServer
import time

@pytest.fixture(scope="function")
def unused_tcp_port():
    """Get an unused TCP port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

@pytest.fixture(scope="function")
def signaling_server(unused_tcp_port):
    server = SignalingServer(port=unused_tcp_port)
    server.start()
    # Allow some time for the server to start
    time.sleep(1)
    yield server
    # No explicit cleanup needed as the server runs in a separate thread

@pytest.fixture
async def websocket_clients(signaling_server, unused_tcp_port):
    """Create two WebSocket clients for testing"""
    client1 = None
    client2 = None
    try:
        uri = f'ws://localhost:{unused_tcp_port}/signaling'
        client1 = await websockets.connect(uri)
        client2 = await websockets.connect(uri)
        yield client1, client2
    finally:
        # Cleanup
        if client1:
            await client1.close()
        if client2:
            await client2.close()

async def send_and_receive(sender, receiver, message):
    """Helper function to send and receive WebSocket messages"""
    await sender.send(json.dumps(message))
    response = await receiver.recv()
    return json.loads(response)


def test_server_initialization():
    """Test basic server initialization"""
    server = SignalingServer()
    assert isinstance(server, SignalingServer)

@pytest.mark.asyncio
async def test_client_connection(signaling_server, unused_tcp_port):
    """Test client connection and count"""
    initial_count = signaling_server.get_client_count()
    assert initial_count == 0

    uri = f'ws://localhost:{unused_tcp_port}/signaling'
    async with websockets.connect(uri) as websocket:
        # Wait briefly for connection to be registered
        await asyncio.sleep(0.1)
        assert signaling_server.get_client_count() == 1

@pytest.mark.asyncio
async def test_signaling_message_exchange(signaling_server, unused_tcp_port):
    """Test sending and receiving signaling messages between clients"""
    uri = f'ws://localhost:{unused_tcp_port}/signaling'
    
    async with websockets.connect(uri) as client1, websockets.connect(uri) as client2:
        # Wait for connections to be established
        await asyncio.sleep(0.1)
        
        # Test SDP offer exchange
        offer_message = {
            "type": "offer",
            "sdp": "test_sdp_offer"
        }
        
        await client1.send(json.dumps(offer_message))
        response = await client2.recv()
        response_data = json.loads(response)
        
        assert response_data["type"] == "offer"
        assert response_data["sdp"] == "test_sdp_offer"

@pytest.mark.asyncio
async def test_broadcast_message(signaling_server, unused_tcp_port):
    """Test broadcasting messages to all clients"""
    uri = f'ws://localhost:{unused_tcp_port}/signaling'
    
    async with websockets.connect(uri) as client1, websockets.connect(uri) as client2:
        # Wait for connections to be established
        await asyncio.sleep(0.1)
        
        test_message = "broadcast test message"
        signaling_server.broadcast_message(test_message)
        
        # Both clients should receive the message
        msg1 = await client1.recv()
        msg2 = await client2.recv()
        
        assert msg1 == test_message
        assert msg2 == test_message
