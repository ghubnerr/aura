"""Train the AffectNet emotion classifier and save a confusion matrix."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from aura.cnn import EmotionModel
from aura.config import CONFIG
from aura.dataset import DatasetProvider


def create_heatmap(true_labels, pred_labels, path: str) -> None:
    matrix = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "confusion_matrix.png"))
    plt.close()


def main() -> None:
    provider = DatasetProvider(augment=True, split=0.7)

    print("Training data distribution")
    provider.label_distr()
    print("=" * 15)

    print("Testing data distribution")
    provider.label_distr(test=True)
    print("=" * 15)

    model = EmotionModel(pretrained=True)
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    true_labels, pred_labels = model.train_loop(
        provider=provider, epochs=3, batch_size=32
    )
    model.test(provider)
    create_heatmap(true_labels, pred_labels, CONFIG.aura_storage)

    print("Saving model...")
    model.save()


if __name__ == "__main__":
    main()
