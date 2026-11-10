"""Bridge from processed faces to emotion embeddings / art conditioning.

Product path: face crop -> emotion embedding -> continuous art engine

    next_frames = step(current_frame, embedding)

When no trained emotion checkpoint is present, a small face-color heuristic
labels mood so the live demo still updates (random ResNet weights do not).
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Optional

import numpy as np
import torch

from aura.cnn import EmotionModel
from aura.config import CONFIG
from aura.inference.emotion_heuristic import heuristic_emotion


class EmotionArtRuntime:
    """Singleton used by the signaling capture loop."""

    _shared: Optional["EmotionArtRuntime"] = None
    _lock = Lock()

    def __init__(self, checkpoint: Optional[str] = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionModel(pretrained=False)
        path = checkpoint or CONFIG.emotion_checkpoint
        self.checkpoint_path = path
        self.use_demo_heuristic = not os.path.isfile(path)
        if not self.use_demo_heuristic:
            self.model.load(path)
            print(f"Emotion runtime: loaded checkpoint {path}")
        else:
            self.model.to(self.device)
            print(
                "Emotion runtime: no checkpoint at "
                f"{path} - using demo face-color heuristic for labels"
            )
        self.model.eval()
        self.last_embedding: Optional[np.ndarray] = None
        self.last_logits: Optional[np.ndarray] = None
        self.last_emotion: Optional[str] = None
        self.label_names = [
            "happy",
            "sad",
            "anger",
            "disgust",
            "fear",
            "neutral",
            "surprise",
            "contempt",
        ]

    @classmethod
    def get_shared(cls) -> "EmotionArtRuntime":
        with cls._lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    def update_from_face(self, face_chw: np.ndarray) -> np.ndarray:
        """Update cached emotion state from a CHW float face crop in [0, 1]."""
        if face_chw.ndim != 3:
            raise ValueError(f"Expected CHW face tensor, got shape {face_chw.shape}")

        if face_chw.shape[0] in (1, 3):
            face_hwc = np.transpose(face_chw, (1, 2, 0))
        else:
            face_hwc = face_chw

        face_u8 = np.clip(face_hwc * 255.0, 0, 255).astype(np.uint8)
        embedding = self.model.embed(face_u8, use_representation=True)
        self.last_embedding = np.asarray(embedding)

        if self.use_demo_heuristic:
            self.last_emotion = heuristic_emotion(face_u8)
            self.last_logits = None
        else:
            logits = self.model.embed(face_u8, use_representation=False)
            if isinstance(logits, torch.Tensor):
                logits_np = logits.detach().cpu().numpy().reshape(-1)
            else:
                logits_np = np.asarray(logits).reshape(-1)
            pred = int(np.argmax(logits_np))
            self.last_logits = logits_np
            self.last_emotion = self.label_names[pred]

        print(
            f"Emotion runtime: {self.last_emotion} "
            f"(embed dim={self.last_embedding.shape[-1]}"
            f"{', demo_heuristic' if self.use_demo_heuristic else ''})"
        )

        try:
            from aura.inference.continuous_art import ContinuousArtEngine

            ContinuousArtEngine.get_shared().set_conditioning(
                self.last_embedding,
                emotion=self.last_emotion,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Continuous conditioning skipped: {exc}")

        return self.last_embedding

    def emotion_message(self) -> str:
        """JSON the browser understands for the emotion status line."""
        import json

        return json.dumps(
            {
                "type": "emotion",
                "emotion": self.last_emotion or "unknown",
                "demo": self.use_demo_heuristic,
            }
        )
