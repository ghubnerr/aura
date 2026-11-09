"""TLS WebSocket proxy in front of the plain Rust signaling server.

Browsers on ``https://`` cannot open ``ws://`` (mixed content). Keep the Rust
server on localhost WS for the streamer, and expose this WSS proxy for the UI:

  python -m aura.webrtc.signaling --bind-ip 127.0.0.1 --port 3030
  python -m aura.webrtc.streamer --signaling-host 127.0.0.1 --signaling-port 3030
  python -m aura.webrtc.wss_proxy --port 3031
  python scripts/serve_ui_https.py

  https://192.168.1.200:8443/?signaling=192.168.1.200:3031
"""

from __future__ import annotations

import argparse
import asyncio
import ssl
from pathlib import Path

import websockets

from aura.certs import default_cert_dir, ensure_self_signed_cert


async def _pump(src, dst) -> None:
    try:
        async for message in src:
            await dst.send(message)
    except websockets.ConnectionClosed:
        pass
    finally:
        try:
            await dst.close()
        except Exception:  # noqa: BLE001
            pass


async def _handler(client, upstream_url: str) -> None:
    async with websockets.connect(upstream_url, max_size=None) as upstream:
        await asyncio.gather(_pump(client, upstream), _pump(upstream, client))


async def run_proxy(host: str, port: int, upstream: str, cert_dir: Path) -> None:
    cert, key = ensure_self_signed_cert(cert_dir)
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))

    async def handler(websocket):
        await _handler(websocket, upstream)

    print(f"WSS proxy on wss://{host}:{port}/ … → {upstream}")
    print(f"TLS cert: {cert}")
    async with websockets.serve(handler, host, port, ssl=ssl_ctx, max_size=None):
        await asyncio.Future()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port",
        type=int,
        default=3031,
        help="Public WSS port browsers connect to",
    )
    parser.add_argument(
        "--upstream",
        default="ws://127.0.0.1:3030/signaling",
        help="Plain Rust signaling URL",
    )
    parser.add_argument("--cert-dir", default=str(default_cert_dir()))
    args = parser.parse_args(argv)
    asyncio.run(run_proxy(args.host, args.port, args.upstream, Path(args.cert_dir)))


if __name__ == "__main__":
    main()
