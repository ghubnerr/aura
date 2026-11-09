"""Demo emotion labels from face color stats (no trained weights required)."""

from __future__ import annotations

import numpy as np


def heuristic_emotion(face_bgr_u8: np.ndarray) -> str:
    """Guess a mood label from mean color / contrast of a BGR face crop."""
    pixels = face_bgr_u8.reshape(-1, 3).astype(np.float32)
    mean = pixels.mean(axis=0)
    b, g, r = (float(x) for x in mean)
    brightness = (r + g + b) / 3.0
    warmth = r - b
    greenness = g - (r + b) / 2.0
    # Luma spatial contrast (ignore channel spread on a flat patch).
    luma = pixels @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
    contrast = float(luma.std())

    if brightness > 145 and warmth > 18:
        return "happy"
    if brightness < 70:
        return "sad"
    if warmth > 35 and contrast > 45:
        return "anger"
    if greenness > 12:
        return "disgust"
    if warmth < -25 and brightness < 110:
        return "fear"
    if contrast > 55 and brightness > 120:
        return "surprise"
    if warmth < -5 and brightness > 100:
        return "contempt"
    return "neutral"
