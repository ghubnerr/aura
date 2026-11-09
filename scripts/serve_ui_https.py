#!/usr/bin/env python3
"""Serve ``ui/static`` over HTTPS so LAN Chrome can access getUserMedia.

Browsers treat http://192.168.x.x as an insecure origin (no camera). This
script creates a self-signed cert (once) and serves the Aura UI on HTTPS.
Accept the cert warning in Chrome, then open:

  https://192.168.1.200:8443/?signaling=192.168.1.200:3030
"""

from __future__ import annotations

import argparse
import http.server
import os
import ssl
import subprocess
import tempfile
from functools import partial
from pathlib import Path


def ensure_cert(cert_dir: Path) -> tuple[Path, Path]:
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert = cert_dir / "aura-dev.pem"
    key = cert_dir / "aura-dev-key.pem"
    if cert.is_file() and key.is_file():
        return cert, key

    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "365",
            "-nodes",
            "-subj",
            "/CN=aura-local",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument(
        "--directory",
        default=str(Path(__file__).resolve().parents[1] / "ui" / "static"),
    )
    parser.add_argument(
        "--cert-dir",
        default=str(Path(tempfile.gettempdir()) / "aura-certs"),
    )
    args = parser.parse_args()

    cert, key = ensure_cert(Path(args.cert_dir))
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=args.directory)
    server = http.server.HTTPServer((args.host, args.port), handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert), keyfile=str(key))
    server.socket = context.wrap_socket(server.socket, server_side=True)

    print(f"Serving {args.directory} on https://{args.host}:{args.port}/")
    print("Self-signed cert — accept the browser warning once.")
    print(
        "Example: "
        f"https://192.168.1.200:{args.port}/?signaling=192.168.1.200:3030"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
