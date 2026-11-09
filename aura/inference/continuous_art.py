"""Continuous emotion-conditioned art generation.

Product contract (not clip-swapping):

    next_frames = backend.step(current_frame, emotion_embedding)

The current RGB frame is the temporal state. The emotion embedding is the
conditioning signal. Backends must produce the *next* frames from that pair so
the WebRTC track is a continuous video, with mood drifting as the embedding
updates — never a hard cut to an unrelated clip.

- ``MorphingStubBackend``: proves continuity on CPU (smooth color/motion drift).
- ``StreamDiffusionBackend`` (later): StreamDiffusionV2 / Krea / Self-Forcing Wan
  with the same ``step(frame, embedding)`` signature on the GPU host.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from aura.config import CONFIG

# Mood palette centers used by the CPU stub (RGB in 0..1).
_EMOTION_RGB = {
    "happy": np.array([1.00, 0.72, 0.05], dtype=np.float32),
    "sad": np.array([0.05, 0.30, 0.70], dtype=np.float32),
    "anger": np.array([0.85, 0.08, 0.05], dtype=np.float32),
    "fear": np.array([0.35, 0.05, 0.55], dtype=np.float32),
    "surprise": np.array([1.00, 0.20, 0.65], dtype=np.float32),
    "disgust": np.array([0.20, 0.55, 0.25], dtype=np.float32),
    "contempt": np.array([0.40, 0.45, 0.40], dtype=np.float32),
    "neutral": np.array([0.35, 0.38, 0.42], dtype=np.float32),
}


class ContinuousArtBackend(Protocol):
    """Predict next art frames from the current image + emotion embedding."""

    def step(
        self,
        current_frame: np.ndarray,
        embedding: np.ndarray,
        n_frames: int = 1,
    ) -> list[np.ndarray]:
        """Return ``n_frames`` RGB uint8 frames continuing from ``current_frame``."""


class MorphingStubBackend:
    """CPU stand-in that keeps temporal continuity across emotion changes.

    Uses the embedding only lightly (norm → motion energy) and an optional
    emotion hint for the target palette. A real diffusion backend will map the
    full embedding into cross-attention / AdaIN conditioning instead.
    """

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._emotion_hint = "neutral"
        self._phase = 0.0
        yy, xx = np.mgrid[0:height, 0:width]
        self._xx = (xx.astype(np.float32) / max(width - 1, 1)) * 2.0 - 1.0
        self._yy = (yy.astype(np.float32) / max(height - 1, 1)) * 2.0 - 1.0

    def set_emotion_hint(self, emotion: Optional[str]) -> None:
        if emotion and emotion in _EMOTION_RGB:
            self._emotion_hint = emotion

    def step(
        self,
        current_frame: np.ndarray,
        embedding: np.ndarray,
        n_frames: int = 1,
    ) -> list[np.ndarray]:
        if current_frame.dtype != np.uint8:
            raise TypeError("current_frame must be uint8 HxWx3 RGB")
        if current_frame.ndim != 3 or current_frame.shape[2] != 3:
            raise ValueError(f"expected HxWx3, got {current_frame.shape}")

        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        energy = float(np.linalg.norm(emb) / max(np.sqrt(emb.size), 1.0))
        energy = float(np.clip(energy / 8.0, 0.15, 1.5))

        target = _EMOTION_RGB.get(self._emotion_hint, _EMOTION_RGB["neutral"])
        frame = current_frame.astype(np.float32) / 255.0
        out: list[np.ndarray] = []

        for _ in range(max(1, n_frames)):
            self._phase += 0.045 * energy
            # Soft blend toward the emotion palette — no hard cut.
            blend = 0.04 + 0.03 * energy
            base = frame * (1.0 - blend) + target.reshape(1, 1, 3) * blend

            wave = 0.5 + 0.5 * np.sin(
                3.0 * self._xx + 2.0 * self._yy + self._phase
            )
            ripple = 0.5 + 0.5 * np.cos(
                5.0 * self._yy - 1.5 * self._xx + 0.7 * self._phase
            )
            field = (0.55 * wave + 0.45 * ripple)[..., None]
            mixed = np.clip(base * (0.65 + 0.55 * field), 0.0, 1.0)

            # Keep a little of the previous pixels for temporal coherence.
            frame = 0.82 * frame + 0.18 * mixed
            out.append((np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8))

        return out


class ContinuousArtEngine:
    """Background loop: condition on embedding, step frames, publish IVF chunks."""

    _shared: Optional["ContinuousArtEngine"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        backend: Optional[MorphingStubBackend] = None,
        ivf_dir: Optional[str] = None,
        fps: float = 24.0,
        chunk_frames: int = 24,
        live_name: str = "live.ivf",
    ) -> None:
        self.backend = backend or MorphingStubBackend()
        self.ivf_dir = Path(ivf_dir or CONFIG.ivf_dir)
        self.ivf_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.chunk_frames = chunk_frames
        self.live_name = live_name

        self._embedding = np.zeros((CONFIG.emotion_embed_dim,), dtype=np.float32)
        self._emotion_hint = "neutral"
        self._frame = self._seed_frame()
        self._cond_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ffmpeg = shutil.which("ffmpeg")

    @classmethod
    def get_shared(cls) -> "ContinuousArtEngine":
        with cls._lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    def _seed_frame(self) -> np.ndarray:
        h, w = self.backend.height, self.backend.width
        target = _EMOTION_RGB["neutral"].reshape(1, 1, 3)
        noise = np.random.rand(h, w, 1).astype(np.float32)
        rgb = np.clip(target * (0.55 + 0.45 * noise), 0.0, 1.0)
        return (rgb * 255.0).astype(np.uint8)

    def set_conditioning(
        self,
        embedding: np.ndarray,
        emotion: Optional[str] = None,
    ) -> None:
        """Update the conditioning signal without resetting temporal state."""
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        with self._cond_lock:
            if emb.size == self._embedding.size:
                self._embedding = emb.copy()
            else:
                # Allow mismatched dims during bring-up; pad/truncate.
                out = np.zeros_like(self._embedding)
                n = min(out.size, emb.size)
                out[:n] = emb[:n]
                self._embedding = out
            if emotion:
                self._emotion_hint = emotion
                if hasattr(self.backend, "set_emotion_hint"):
                    self.backend.set_emotion_hint(emotion)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self._ffmpeg:
            print("ContinuousArtEngine: ffmpeg not found; cannot encode IVF chunks")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="aura-continuous-art",
            daemon=True,
        )
        self._thread.start()
        print(
            f"ContinuousArtEngine started "
            f"({self.fps} fps, chunk={self.chunk_frames} frames)"
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        buffer: list[np.ndarray] = []
        period = 1.0 / max(self.fps, 1.0)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._cond_lock:
                emb = self._embedding.copy()
            frames = self.backend.step(self._frame, emb, n_frames=1)
            self._frame = frames[-1]
            buffer.append(self._frame)

            if len(buffer) >= self.chunk_frames:
                try:
                    self._publish_chunk(buffer)
                except Exception as exc:  # noqa: BLE001 - keep loop alive
                    print(f"ContinuousArtEngine publish failed: {exc}")
                buffer = []

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))

    def _publish_chunk(self, frames: Sequence[np.ndarray]) -> None:
        """Encode a short continuous chunk and atomically replace live.ivf."""
        if not frames:
            return
        h, w, _ = frames[0].shape
        dest = self.ivf_dir / self.live_name
        tmp = self.ivf_dir / f".{self.live_name}.encode.ivf"
        raw = self.ivf_dir / f".{self.live_name}.raw"

        # Concatenate RGB24 frames for ffmpeg rawvideo.
        blob = b"".join(frame.tobytes() for frame in frames)
        raw.write_bytes(blob)

        cmd = [
            self._ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(self.fps),
            "-i",
            str(raw),
            "-c:v",
            "libvpx",
            "-b:v",
            "1M",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-an",
            str(tmp),
        ]
        subprocess.run(cmd, check=True)

        last_err: Optional[Exception] = None
        for attempt in range(12):
            try:
                os.replace(tmp, dest)
                break
            except PermissionError as exc:
                last_err = exc
                time.sleep(0.05 * (attempt + 1))
        else:
            if last_err:
                raise last_err

        try:
            os.utime(dest, None)
        except OSError:
            pass
        try:
            raw.unlink(missing_ok=True)
        except OSError:
            pass
