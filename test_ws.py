import asyncio
import websockets
import ssl
import logging
import sys
import json

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

async def test_connection():
    uri = "wss://onyx.cs.fiu.edu:9443/ws"
    logging.info(f"Attempting to connect to {uri}")
    
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        logging.debug("Created SSL context")
        
        async with websockets.connect(
            uri,
            ssl=ssl_context
        ) as websocket:
            logging.info("Connected successfully")
            
            # Create a proper signaling message
            test_message = {
                "type": "triggerimagecapture"
            }
                    
            logging.debug("Sending test message")
            await websocket.send(json.dumps(test_message))
            
            logging.debug("Waiting for response")
            response = await websocket.recv()
            logging.info(f"Received response: {response}")
            
            # Parse the response
            try:
                response_json = json.loads(response)
                logging.debug(f"Parsed response: {response_json}")
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse response as JSON: {e}")
            
    except websockets.exceptions.InvalidStatus as e:
        logging.error(f"Invalid status code: {e.response.status_code}")
    except websockets.exceptions.InvalidHandshake as e:
        logging.error(f"Invalid handshake: {str(e)}")
    except websockets.exceptions.ConnectionClosed as e:
        logging.error(f"Connection closed: code={e.code}, reason={e.reason}")
    except Exception as e:
        logging.error(f"Connection failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_connection())
