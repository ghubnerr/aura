import asyncio
import websockets
import logging
import json
from typing import Dict, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SignalingServer:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()

    async def register(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client"""
        self.clients.add(websocket)
        logging.info(f"New client connected. Total clients: {len(self.clients)}")

    async def unregister(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client"""
        self.clients.remove(websocket)
        logging.info(f"Client disconnected. Total clients: {len(self.clients)}")

    async def broadcast(self, message: str, sender: websockets.WebSocketServerProtocol):
        """Broadcast message to all clients except sender"""
        for client in self.clients:
            if client != sender:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    await self.unregister(client)

    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle individual client connections"""
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    # Parse message to verify it's valid JSON
                    data = json.loads(message)
                    logging.info(f"Received message: {message}")
                    await self.broadcast(message, websocket)
                except json.JSONDecodeError:
                    logging.error("Invalid JSON message received")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

    async def start(self):
        """Start the signaling server"""
        server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port
        )
        logging.info(f"Signaling server running on ws://{self.host}:{self.port}")
        await server.wait_closed()

def main():
    """Main function to run the server"""
    server = SignalingServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logging.info("Server stopped by user")

if __name__ == "__main__":
    main()