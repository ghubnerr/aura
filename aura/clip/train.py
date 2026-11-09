"""Train projection heads that align emotion embeddings with art frames.

This is the packaged version of the prototype in ``notebooks/clip.ipynb``.
Only the projection heads are trained; both backbones stay frozen.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aura.clip.model import CLIPImageEncoder, EmotionEncoderWithProjection
from aura.config import CONFIG


class EmotionAwareInfoNCE(torch.nn.Module):
    """InfoNCE where same-emotion samples within a batch are positives."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        emotion_z: torch.Tensor,
        image_z: torch.Tensor,
        emotion_labels: torch.Tensor,
    ) -> torch.Tensor:
        emotion_z = F.normalize(emotion_z, dim=-1)
        image_z = F.normalize(image_z, dim=-1)
        logits = emotion_z @ image_z.T / self.temperature

        labels = emotion_labels.unsqueeze(0)
        positive_mask = labels == labels.T
        positive_mask.fill_diagonal_(False)

        # Fall back to standard diagonal positives when a class appears once.
        if positive_mask.sum() == 0:
            targets = torch.arange(logits.size(0), device=logits.device)
            return F.cross_entropy(logits, targets)

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(
            dim=1
        ).clamp_min(1)
        return -mean_log_prob_pos.mean()


def train_projection_heads(
    dataset: Dataset,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[CLIPImageEncoder, EmotionEncoderWithProjection]:
    """Train CLIP/emotion projection heads on a contrastive dataset.

    The dataset must yield ``(face, frame, emotion_label)`` tensors.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = checkpoint_dir or os.path.join(CONFIG.aura_storage, "clip")
    os.makedirs(checkpoint_dir, exist_ok=True)

    image_encoder = CLIPImageEncoder(device=device).to(device)
    emotion_encoder = EmotionEncoderWithProjection(device=device).to(device)
    criterion = EmotionAwareInfoNCE()
    optimizer = torch.optim.Adam(
        list(image_encoder.projection.parameters())
        + list(emotion_encoder.projection_head.parameters()),
        lr=lr,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    best_loss = float("inf")

    for epoch in range(epochs):
        image_encoder.train()
        emotion_encoder.train()
        running = 0.0
        steps = 0
        for faces, frames, labels in loader:
            faces = faces.to(device)
            frames = frames.to(device)
            labels = labels.to(device)

            emotion_z = emotion_encoder(faces)
            image_z = image_encoder(frames)
            loss = criterion(emotion_z, image_z, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            steps += 1

        epoch_loss = running / max(steps, 1)
        print(f"Epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "image_projection": image_encoder.projection.state_dict(),
                    "emotion_projection": emotion_encoder.projection_head.state_dict(),
                    "loss": best_loss,
                },
                os.path.join(checkpoint_dir, "best_model.pth"),
            )

    return image_encoder, emotion_encoder


def _main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)
    raise SystemExit(
        "Provide a contrastive Dataset via train_projection_heads(); "
        f"CLI scaffolding only (epochs={args.epochs})."
    )


if __name__ == "__main__":
    _main()
