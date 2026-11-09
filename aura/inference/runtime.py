"""Bridge from processed faces to emotion embeddings / art conditioning.

Until StyleGAN-V or StreamDiffusionV2 weights are wired in, this runtime only
extracts and caches the emotion embedding so the WebRTC path stays end-to-end
testable.
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Optional

import numpy as np
import torch

from aura.cnn import EmotionModel
from aura.config import CONFIG


class EmotionArtRuntime:
    """Singleton used by the signaling capture loop."""

    _shared: Optional["EmotionArtRuntime"] = None
    _lock = Lock()

    def __init__(self, checkpoint: Optional[str] = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionModel(pretrained=False)
        path = checkpoint or CONFIG.emotion_checkpoint
        if os.path.isfile(path):
            self.model.load(path)
        else:
            # Random weights keep the live path exercisable before training.
            self.model.to(self.device)
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
        # ProcessingPipeline returns CHW float; embed expects HWC.
        if face_chw.ndim != 3:
            raise ValueError(f"Expected CHW face tensor, got shape {face_chw.shape}")

        if face_chw.shape[0] in (1, 3):
            face_hwc = np.transpose(face_chw, (1, 2, 0))
        else:
            face_hwc = face_chw

        face_u8 = np.clip(face_hwc * 255.0, 0, 255).astype(np.uint8)
        # Faces from OpenCV are BGR; EmotionModel.embed converts BGR→RGB.
        embedding = self.model.embed(face_u8, use_representation=True)
        logits = self.model.embed(face_u8, use_representation=False)
        if isinstance(logits, torch.Tensor):
            logits_np = logits.detach().cpu().numpy().reshape(-1)
        else:
            logits_np = np.asarray(logits).reshape(-1)

        pred = int(np.argmax(logits_np))
        self.last_embedding = np.asarray(embedding)
        self.last_logits = logits_np
        self.last_emotion = self.label_names[pred]
        print(
            f"Emotion runtime: {self.last_emotion} "
            f"(embed dim={self.last_embedding.shape[-1]})"
        )
        return self.last_embedding
