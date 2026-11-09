"""Shared utility helpers for Aura."""

from __future__ import annotations

import base64
import hashlib

import numpy as np


def hash_image(img: np.ndarray, length: int = 10) -> str:
    """Return a short, stable hash of an image array.

    Args:
        img: Image array of any shape.
        length: Number of base32 characters to keep.

    Returns:
        Truncated base32 SHA-1 digest of the raw image bytes.
    """
    sha1_hash = hashlib.sha1(img.tobytes()).digest()
    encoded = base64.b32encode(sha1_hash).decode("utf-8")
    return encoded[:length]
