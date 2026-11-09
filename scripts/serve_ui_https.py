#!/usr/bin/env python3
"""Serve ``ui/static`` over HTTPS so LAN Chrome can access getUserMedia.

Pair with WSS signaling (``python -m aura.webrtc.signaling --tls``) so the
page is not mixed-content blocked:

  https://192.168.1.200:8443/?signaling=192.168.1.200:3030
"""

from __future__ import annotations

import argparse
import http.server
import ssl
from functools import partial
from pathlib import Path

from aura.certs import default_cert_dir, ensure_self_signed_cert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument(
        "--directory",
        default=str(Path(__file__).resolve().parents[1] / "ui" / "static"),
    )
    parser.add_argument("--cert-dir", default=str(default_cert_dir()))
    args = parser.parse_args()

    cert, key = ensure_self_signed_cert(Path(args.cert_dir))
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=args.directory)
    server = http.server.HTTPServer((args.host, args.port), handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert), keyfile=str(key))
    server.socket = context.wrap_socket(server.socket, server_side=True)

    print(f"Serving {args.directory} on https://{args.host}:{args.port}/")
    print(f"TLS cert: {cert}")
    print("Accept the self-signed cert warning once in the browser.")
    print(
        "Example: "
        f"https://192.168.1.200:{args.port}/?signaling=192.168.1.200:3030"
    )
    print("Signaling must also use --tls with the same cert directory.")
    server.serve_forever()


if __name__ == "__main__":
    main()
