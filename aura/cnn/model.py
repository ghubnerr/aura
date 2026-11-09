"""ResNet50 emotion classifier / feature encoder."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torchvision import models, transforms
from tqdm import tqdm

from aura.config import CONFIG


class EmotionModel(nn.Module):
    """8-way AffectNet emotion classifier with optional GAP embeddings.

    The final FC maps 2048 → 8. Call :meth:`embed` with
    ``use_representation=True`` to get the 2048-d emotion vector used by CLIP
    alignment and live conditioning.
    """

    NUM_CLASSES = 8
    EMBED_DIM = 2048

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, self.NUM_CLASSES)
        self.fc = self.backbone.fc

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=2, gamma=0.1
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape (batch, 8)."""
        return self.backbone(x)

    def _prepare_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        skip_transform: bool,
    ) -> torch.Tensor:
        """Convert an image to a batched float tensor on ``self.device``."""
        if skip_transform:
            if isinstance(image, np.ndarray):
                tensor = torch.from_numpy(image)
            else:
                tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            return tensor.to(self.device)

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        # OpenCV paths are BGR; convert to RGB for ImageNet-normalized weights.
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1].copy()

        if isinstance(image, np.ndarray) and image.dtype == np.uint8:
            pil_image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            clipped = np.clip(image, 0.0, 1.0)
            pil_image = Image.fromarray((clipped * 255).astype(np.uint8))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor.to(self.device)

    def embed(
        self,
        image: Union[np.ndarray, torch.Tensor],
        use_representation: bool = False,
        skip_transform: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Extract GAP features or class logits from one image.

        Args:
            image: HxWx3 uint8/float array, or a CHW / BCHW tensor.
            use_representation: If True, return the 2048-d GAP vector.
            skip_transform: If True, assume ``image`` is already normalized.

        Returns:
            ``np.ndarray`` of shape (2048,) when ``use_representation`` else
            logits ``torch.Tensor`` of shape (1, 8).
        """
        batch = self._prepare_image(image, skip_transform=skip_transform)

        if not use_representation:
            self.eval()
            with torch.no_grad():
                return self(batch)

        representation: dict[str, torch.Tensor] = {}

        def _hook(_module, _inputs, output: torch.Tensor) -> None:
            representation["gap"] = output

        handle = self.backbone.avgpool.register_forward_hook(_hook)
        self.eval()
        try:
            with torch.no_grad():
                self(batch)
        finally:
            handle.remove()

        gap = representation["gap"].squeeze().detach().cpu().numpy()
        return gap

    def save(self, path: Optional[str] = None) -> bool:
        """Persist weights to ``path`` or the default Aura storage location."""
        save_path = path or CONFIG.emotion_checkpoint
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(self.state_dict(), save_path)
        return True

    def load(self, path: str) -> bool:
        """Load weights from ``path``."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict)
        self.to(self.device)
        return True

    def train_loop(
        self, provider, epochs: int, batch_size: int
    ) -> Tuple[List[int], List[int]]:
        """Train on ``provider`` train split; return true/pred label lists."""
        self.to(self.device)
        true_labels: List[int] = []
        pred_labels: List[int] = []

        self.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            batches = provider.get_next_image_batch(batch_size, source="train")
            progress = tqdm(batches, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

            for batch in progress:
                images = []
                labels = []
                for image, label, _emotion in batch:
                    # Convert BGR → RGB before ImageNet normalization.
                    rgb = image[:, :, ::-1] if image.ndim == 3 else image
                    images.append(self.transform(Image.fromarray(rgb)))
                    labels.append(label)

                image_batch = torch.stack(images).to(self.device)
                label_batch = torch.tensor(labels, device=self.device)

                self.optimizer.zero_grad()
                outputs = self(image_batch)
                loss = self.criterion(outputs, label_batch)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * image_batch.size(0)
                _, predicted = outputs.max(1)
                true_labels.extend(label_batch.detach().cpu().tolist())
                pred_labels.extend(predicted.detach().cpu().tolist())
                total += label_batch.size(0)
                correct += predicted.eq(label_batch).sum().item()
                progress.set_postfix(
                    loss=loss.item(), accuracy=100.0 * correct / max(total, 1)
                )

            print(
                f"Epoch {epoch + 1}, "
                f"Loss: {running_loss / max(total, 1):.4f}, "
                f"Accuracy: {100.0 * correct / max(total, 1):.2f}%"
            )
            self.scheduler.step()

        return true_labels, pred_labels

    def test(self, provider) -> Tuple[List[int], List[int]]:
        """Evaluate on the provider test split."""
        self.to(self.device)
        self.eval()
        true_labels: List[int] = []
        pred_labels: List[int] = []
        correct = 0
        total = 0
        running_loss = 0.0
        correct_per_class = [0] * self.NUM_CLASSES
        total_per_class = [0] * self.NUM_CLASSES

        with torch.no_grad():
            for batch in provider.get_next_image_batch(
                len(provider.test), source="test"
            ):
                images = []
                labels = []
                for image, label, _emotion in batch:
                    rgb = image[:, :, ::-1] if image.ndim == 3 else image
                    images.append(self.transform(Image.fromarray(rgb)))
                    labels.append(label)

                image_batch = torch.stack(images).to(self.device)
                label_batch = torch.tensor(labels, device=self.device)
                outputs = self(image_batch)
                loss = self.criterion(outputs, label_batch)
                running_loss += loss.item() * image_batch.size(0)

                _, predicted = outputs.max(1)
                true_labels.extend(label_batch.cpu().tolist())
                pred_labels.extend(predicted.cpu().tolist())
                total += label_batch.size(0)
                correct += predicted.eq(label_batch).sum().item()

                for label_i, pred_i in zip(
                    label_batch.tolist(), predicted.tolist()
                ):
                    total_per_class[label_i] += 1
                    if pred_i == label_i:
                        correct_per_class[label_i] += 1

        print(
            f"Validation Results, Loss: {running_loss / max(total, 1):.4f}, "
            f"Accuracy: {100.0 * correct / max(total, 1):.2f}%"
        )
        for label in range(self.NUM_CLASSES):
            if total_per_class[label] == 0:
                print(f"Label {label}: No samples in validation set.")
                continue
            acc = 100.0 * correct_per_class[label] / total_per_class[label]
            print(
                f"Label {label}: Accuracy {acc:.2f}% "
                f"({correct_per_class[label]}/{total_per_class[label]})"
            )
        return true_labels, pred_labels


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = EmotionModel(pretrained=True)
    model.to(model.device)
    model.eval()

    test_image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    features = model.embed(test_image, use_representation=True)
    assert features.shape == (EmotionModel.EMBED_DIM,)
    print("Smoke test passed:", features.shape)
