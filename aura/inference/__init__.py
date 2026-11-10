"""Live emotion -> art runtime hooks."""

from __future__ import annotations

__all__ = [
    "ContinuousArtEngine",
    "DiffusersTurboBackend",
    "EmotionArtGenerator",
    "EmotionArtRuntime",
    "MorphingStubBackend",
    "create_art_backend",
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
    if name in {"DiffusersTurboBackend", "create_art_backend"}:
        from aura.inference import diffusers_stream

        return getattr(diffusers_stream, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
