"""Diffusion backend factory / prompt mapping tests (CPU-only)."""

from __future__ import annotations

import os

from aura.inference.diffusers_stream import EMOTION_PROMPTS, create_art_backend


def test_emotion_prompts_cover_labels():
    for key in (
        "happy",
        "sad",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "contempt",
        "neutral",
    ):
        assert key in EMOTION_PROMPTS
        assert "abstract" in EMOTION_PROMPTS[key]
        assert "no people" in EMOTION_PROMPTS[key]


def test_create_backend_morphing(monkeypatch):
    monkeypatch.setenv("AURA_ART_BACKEND", "morphing")
    backend = create_art_backend()
    assert type(backend).__name__ == "MorphingStubBackend"


def test_create_backend_unknown_falls_back(monkeypatch):
    monkeypatch.setenv("AURA_ART_BACKEND", "not-a-real-backend")
    backend = create_art_backend()
    assert type(backend).__name__ == "MorphingStubBackend"
