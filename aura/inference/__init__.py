"""Live emotion -> art runtime hooks."""

from __future__ import annotations

__all__ = [
    "ContinuousArtEngine",
    "EmotionArtGenerator",
    "EmotionArtRuntime",
    "MorphingStubBackend",
]


def __getattr__(name: str):
    if name == "EmotionArtGenerator":
        from aura.inference.art_generator import EmotionArtGenerator

        return EmotionArtGenerator
    if name == "EmotionArtRuntime":
        from aura.inference.runtime import EmotionArtRuntime

        return EmotionArtRuntime
    if name in {"ContinuousArtEngine", "MorphingStubBackend"}:
        from aura.inference import continuous_art

        return getattr(continuous_art, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
