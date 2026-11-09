"""Live emotion → art runtime hooks."""

from __future__ import annotations

__all__ = ["EmotionArtGenerator", "EmotionArtRuntime"]


def __getattr__(name: str):
    if name == "EmotionArtGenerator":
        from aura.inference.art_generator import EmotionArtGenerator

        return EmotionArtGenerator
    if name == "EmotionArtRuntime":
        from aura.inference.runtime import EmotionArtRuntime

        return EmotionArtRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
