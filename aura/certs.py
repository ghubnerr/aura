"""Self-signed TLS material shared by HTTPS UI and WSS signaling."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple


def default_cert_dir() -> Path:
    override = os.environ.get("AURA_TLS_DIR")
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "aura-certs"


def ensure_self_signed_cert(
    cert_dir: Path | None = None,
) -> Tuple[Path, Path]:
    """Return ``(cert.pem, key.pem)``, creating them with openssl if needed."""
    root = cert_dir or default_cert_dir()
    root.mkdir(parents=True, exist_ok=True)
    cert = root / "aura-dev.pem"
    key = root / "aura-dev-key.pem"
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
