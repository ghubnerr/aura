"""Continuous art backend contract tests."""

from __future__ import annotations

import numpy as np

from aura.inference.continuous_art import MorphingStubBackend


def test_step_keeps_shape_and_continuity():
    backend = MorphingStubBackend(width=64, height=48)
    frame = np.full((48, 64, 3), 40, dtype=np.uint8)
    embedding = np.ones((2048,), dtype=np.float32)

    out1 = backend.step(frame, embedding, n_frames=3)
    assert len(out1) == 3
    assert out1[0].shape == (48, 64, 3)
    assert out1[0].dtype == np.uint8

    # Emotion change must not hard-reset to an unrelated image: next frames
    # should stay close to the previous temporal state.
    backend.set_emotion_hint("anger")
    out2 = backend.step(out1[-1], embedding * 1.2, n_frames=1)[0]
    delta = np.mean(np.abs(out2.astype(np.int16) - out1[-1].astype(np.int16)))
    assert delta < 40


def test_set_conditioning_updates_engine_without_reset(tmp_path):
    from aura.inference.continuous_art import ContinuousArtEngine, MorphingStubBackend

    engine = ContinuousArtEngine(
        backend=MorphingStubBackend(width=64, height=48),
        ivf_dir=str(tmp_path),
        fps=12,
        chunk_frames=4,
    )
    before = engine._frame.copy()
    engine.set_conditioning(np.random.randn(2048).astype(np.float32), emotion="happy")
    assert engine._emotion_hint == "happy"
    # Temporal state preserved on conditioning updates.
    assert np.array_equal(engine._frame, before)
