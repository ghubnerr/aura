"""Publish emotion-conditioned abstract art into the IVF watch directory.

MVP path: swap prebuilt VP8 stubs (palette / motion keyed by emotion). Later
this module can call StreamDiffusionV2 / Krea Realtime and write live.ivf.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Optional

from aura.config import CONFIG

# Package-relative defaults; overridden by AURA_EMOTION_IVF_DIR.
_REPO_SAMPLES = (
    Path(__file__).resolve().parents[2] / "docs" / "samples" / "emotions"
)

EMOTION_ALIASES = {
    "joy": "happy",
    "happiness": "happy",
    "sadness": "sad",
    "angry": "anger",
}


class EmotionArtGenerator:
    """Map an emotion label to a streaming IVF clip."""

    def __init__(
        self,
        ivf_dir: Optional[str] = None,
        sample_dir: Optional[str] = None,
        live_name: str = "live.ivf",
    ) -> None:
        self.ivf_dir = Path(ivf_dir or CONFIG.ivf_dir)
        self.sample_dir = Path(
            sample_dir
            or os.environ.get("AURA_EMOTION_IVF_DIR", str(_REPO_SAMPLES))
        )
        self.live_name = live_name
        self.ivf_dir.mkdir(parents=True, exist_ok=True)
        self.last_emotion: Optional[str] = None
        self.last_path: Optional[Path] = None

    def resolve_emotion(self, emotion: str) -> str:
        key = emotion.lower().strip()
        key = EMOTION_ALIASES.get(key, key)
        candidate = self.sample_dir / f"{key}.ivf"
        if candidate.is_file():
            return key
        return "neutral"

    def publish(self, emotion: str, force: bool = False) -> Path:
        """Copy the emotion stub to ``live.ivf`` and bump mtime for the watcher."""
        resolved = self.resolve_emotion(emotion)
        if not force and resolved == self.last_emotion and self.last_path:
            # Still touch so a late-joining streamer can pick it up.
            os.utime(self.last_path, None)
            return self.last_path

        source = self.sample_dir / f"{resolved}.ivf"
        if not source.is_file():
            fallback = Path("docs/samples/aura_stub.ivf")
            if not fallback.is_file():
                raise FileNotFoundError(
                    f"No IVF stub for emotion '{emotion}' in {self.sample_dir}"
                )
            source = fallback

        dest = self.ivf_dir / self.live_name
        tmp = self.ivf_dir / f".{self.live_name}.tmp"
        shutil.copy2(source, tmp)
        os.replace(tmp, dest)
        now = time.time()
        os.utime(dest, (now, now))

        self.last_emotion = resolved
        self.last_path = dest
        print(f"Art generator: published {resolved} → {dest}")
        return dest
