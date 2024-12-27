import pytest
import asyncio
import websockets
import json
from aura.webrtc import SignalingServer
import sys

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

    async def is_server_ready():
        uri = f'ws://localhost:{unused_tcp_port}/signaling'
        for _ in range(10):
            try:
                async with websockets.connect(uri):
                    return True  
            except (ConnectionRefusedError, OSError):
                await asyncio.sleep(0.1)  
        return False 

    async def wait_for_server():
        return await asyncio.wait_for(is_server_ready(), timeout=30.0)

    if not asyncio.run(wait_for_server()):
        raise RuntimeError("Server did not start within the timeout period")

    yield server

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
        await asyncio.sleep(0.1)
        assert signaling_server.get_client_count() == 1

@pytest.mark.asyncio
async def test_signaling_message_exchange(signaling_server, unused_tcp_port):
    """Test sending and receiving signaling messages between clients"""
    uri = f'ws://localhost:{unused_tcp_port}/signaling'
    
    async with websockets.connect(uri) as client1, websockets.connect(uri) as client2:
        await asyncio.sleep(0.1)
        
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
        await asyncio.sleep(0.1)
        
        test_message = "broadcast test message"
        signaling_server.broadcast_message(test_message)
        
        msg1 = await client1.recv()
        msg2 = await client2.recv()
        
        assert msg1 == test_message
        assert msg2 == test_message

@pytest.mark.asyncio
async def test_disconnect_client(signaling_server, unused_tcp_port):
    """Test disconnecting a client"""
    uri = f'ws://localhost:{unused_tcp_port}/signaling'
    
    async with websockets.connect(uri) as client:
        await asyncio.sleep(0.1)
        
        connected_clients = signaling_server.get_connected_clients()
        assert len(connected_clients) == 1
        client_id = connected_clients[0]
        
        assert signaling_server.disconnect_client(client_id) == True
        await asyncio.sleep(0.1)
        
        assert signaling_server.get_client_count() == 0
        assert signaling_server.disconnect_client(client_id) == False

@pytest.mark.asyncio
async def test_send_to_client(signaling_server, unused_tcp_port):
    """Test sending messages to specific clients"""
    uri = f'ws://localhost:{unused_tcp_port}/signaling'
    
    async with websockets.connect(uri) as client1, websockets.connect(uri) as client2:
        await asyncio.sleep(0.1)
        
        clients = signaling_server.get_connected_clients()
        client_id = clients[0]
        
        test_message = "test message"
        assert signaling_server.send_to_client(client_id, test_message) == True
        
        assert signaling_server.send_to_client("invalid_id", test_message) == False
        
        received_msg = await client1.recv()
        assert received_msg == test_message

@pytest.mark.asyncio
async def test_server_status(signaling_server, unused_tcp_port):
    """Test server status information"""
    uri = f'ws://localhost:{unused_tcp_port}/signaling'
    
    status = signaling_server.get_server_status()
    assert isinstance(status, dict)
    assert "ip" in status
    assert "port" in status
    assert "connected_clients" in status
    assert status["port"] == str(unused_tcp_port)
    
    async with websockets.connect(uri) as client:
        await asyncio.sleep(0.1)
        updated_status = signaling_server.get_server_status()
        assert updated_status["connected_clients"] == "1"

@pytest.mark.asyncio
async def test_client_capacity(signaling_server, unused_tcp_port):
    """Test server capacity handling"""
    uri = f'ws://localhost:{unused_tcp_port}/signaling'
    
    assert signaling_server.is_at_capacity() == False
    
    async with websockets.connect(uri) as client1:
        await asyncio.sleep(0.1)
        assert signaling_server.is_at_capacity() == False
        
        async with websockets.connect(uri) as client2:
            await asyncio.sleep(0.1)
            assert signaling_server.is_at_capacity() == True
