"""Demo emotion heuristic tests."""

from __future__ import annotations

import numpy as np

from aura.inference.emotion_heuristic import heuristic_emotion


def test_warm_bright_face_is_happy():
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    # Uniform warm/bright BGR — low contrast so anger branch does not win.
    face[:] = (60, 140, 210)
    assert heuristic_emotion(face) == "happy"


def test_dark_face_is_sad():
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    face[:] = (30, 25, 20)
    assert heuristic_emotion(face) == "sad"
