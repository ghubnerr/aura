"""Frozen backbones + projection heads for emotion–video contrastive learning."""

from __future__ import annotations

from typing import Optional, Union

import clip
import numpy as np
import torch
import torch.nn as nn

from aura.cnn import EmotionModel
from aura.config import CONFIG


class CLIPImageEncoder(nn.Module):
    """Project frozen OpenAI CLIP image features into the shared Aura space."""

    def __init__(
        self,
        projection_dim: Optional[int] = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        dim = projection_dim or CONFIG.clip_projection_dim
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(512, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a batch of image frames to the shared projection space."""
        with torch.no_grad():
            features = self.clip_model.encode_image(frames).float()
        return self.projection(features)


class EmotionEncoderWithProjection(nn.Module):
    """Project frozen emotion CNN GAP features into the shared Aura space."""

    def __init__(
        self,
        projection_dim: Optional[int] = None,
        device: str = "cuda",
        emotion_model_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        dim = projection_dim or CONFIG.clip_projection_dim
        checkpoint = emotion_model_path or CONFIG.emotion_checkpoint

        self.base_encoder = EmotionModel(pretrained=False).to(self.device)
        if os_path_exists(checkpoint):
            self.base_encoder.load(checkpoint)
        for param in self.base_encoder.parameters():
            param.requires_grad = False

        self.projection_head = nn.Sequential(
            nn.Linear(EmotionModel.EMBED_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, faces: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Encode face images to the shared projection space."""
        if isinstance(faces, torch.Tensor):
            faces_np = faces.detach().cpu().numpy()
        else:
            faces_np = faces

        if faces_np.ndim == 3:
            faces_np = np.expand_dims(faces_np, axis=0)

        embeddings = []
        for face in faces_np:
            # CHW float tensors from ProcessingPipeline need HWC uint8/float.
            if face.ndim == 3 and face.shape[0] in (1, 3):
                face = np.transpose(face, (1, 2, 0))
            embedding = self.base_encoder.embed(face, use_representation=True)
            embeddings.append(embedding)

        batch = torch.from_numpy(np.stack(embeddings)).float().to(self.device)
        return self.projection_head(batch)


def os_path_exists(path: str) -> bool:
    """Local import wrapper so model.py stays importable without side effects."""
    import os

    return os.path.isfile(path)
