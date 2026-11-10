"""Emotion → IVF publisher smoke tests."""

from __future__ import annotations

from pathlib import Path

from aura.inference.art_generator import EmotionArtGenerator


def test_publish_copies_emotion_stub(tmp_path: Path):
    samples = Path("docs/samples/emotions")
    assert (samples / "happy.ivf").is_file()

    gen = EmotionArtGenerator(ivf_dir=str(tmp_path), sample_dir=str(samples))
    dest = gen.publish("happy", force=True)
    assert dest.is_file()
    assert dest.name == "live.ivf"
    assert dest.stat().st_size == (samples / "happy.ivf").stat().st_size
    assert gen.last_emotion == "happy"


def test_alias_joy_maps_to_happy(tmp_path: Path):
    samples = Path("docs/samples/emotions")
    gen = EmotionArtGenerator(ivf_dir=str(tmp_path), sample_dir=str(samples))
    assert gen.resolve_emotion("joy") == "happy"
