"""Aura: real-time emotion-aligned abstract art over WebRTC."""

from __future__ import annotations

from aura.config import CONFIG, load_config

# Native PyO3 extension built by maturin (`aura.aura`).
from aura.aura import SignalingServer, VideoStreamer

__all__ = [
    "CONFIG",
    "SignalingServer",
    "VideoStreamer",
    "load_config",
]

__version__ = "0.1.0"
