"""Train the AffectNet emotion classifier and save a checkpoint for live Aura.

Usage (Gabes-PC / any CUDA host)::

    python -m aura.cnn.train --epochs 5 --batch-size 32

Requires Kaggle credentials for ``kagglehub`` (AffectNet mirror
``noamsegal/affectnet-training-data``). Writes:

    <STORAGE_PATH>/aura_storage/aura_emotion_classifier.pth
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from aura.cnn import EmotionModel
from aura.config import CONFIG
from aura.dataset import DatasetProvider

LABEL_NAMES = [
    "happy",
    "sad",
    "anger",
    "disgust",
    "fear",
    "neutral",
    "surprise",
    "contempt",
]


def create_heatmap(true_labels, pred_labels, path: str) -> None:
    matrix = confusion_matrix(true_labels, pred_labels, labels=list(range(8)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs(path, exist_ok=True)
    out = os.path.join(path, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--split", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=2000,
        help="Cap original images per emotion before augmentation",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable on-the-fly flip/rotate/brightness variants",
    )
    parser.add_argument(
        "--checkpoint",
        default=CONFIG.emotion_checkpoint,
        help="Where to write the trained weights",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional existing checkpoint to fine-tune from",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Only train the final FC (faster smoke run)",
    )
    args = parser.parse_args(argv)

    _set_seed(args.seed)

    # Allow DatasetProvider cutoff override without editing the module.
    import aura.dataset.dataset_provider as dp

    dp.CLASS_CUTOFF = max(1, args.max_per_class)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading AffectNet via kagglehub (first run downloads)...")
    provider = DatasetProvider(
        target_size=(224, 224),
        augment=not args.no_augment,
        split=args.split,
    )
    # Resize every stored crop so ResNet always sees 224x224.
    provider.train = [
        (provider._resize_image(img), label, name)
        for img, label, name in provider.train
    ]
    provider.test = [
        (provider._resize_image(img), label, name)
        for img, label, name in provider.test
    ]
    provider.dataset = provider.train + provider.test

    print(f"Train samples: {len(provider.train)} | Test samples: {len(provider.test)}")
    print("Training distribution")
    provider.label_distr()
    print("Test distribution")
    provider.label_distr(test=True)

    model = EmotionModel(pretrained=True)
    model.device = device

    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        print("Frozen backbone; training FC only")
    else:
        # Fine-tune layer4 + head (rest stays ImageNet-initialized).
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
        print("Training layer4 + FC")

    model.optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    model.scheduler = torch.optim.lr_scheduler.StepLR(
        model.optimizer, step_size=max(1, args.epochs // 2), gamma=0.5
    )

    if args.resume and os.path.isfile(args.resume):
        model.load(args.resume)
        print(f"Resumed from {args.resume}")

    model.to(device)

    best_acc = -1.0
    best_path = args.checkpoint
    history = []

    for epoch in range(args.epochs):
        random.shuffle(provider.train)
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        model.train_loop(provider=provider, epochs=1, batch_size=args.batch_size)
        true_labels, pred_labels = model.test(provider)
        correct = sum(int(t == p) for t, p in zip(true_labels, pred_labels))
        acc = 100.0 * correct / max(len(true_labels), 1)
        history.append({"epoch": epoch + 1, "val_acc": acc})
        print(f"Val accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            model.save(best_path)
            print(f"New best checkpoint -> {best_path} ({best_acc:.2f}%)")

    # Final report + confusion matrix from last eval.
    true_labels, pred_labels = model.test(provider)
    print(
        classification_report(
            true_labels,
            pred_labels,
            labels=list(range(8)),
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )
    create_heatmap(true_labels, pred_labels, CONFIG.aura_storage)

    metrics_path = Path(CONFIG.aura_storage) / "emotion_train_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_acc": best_acc,
                "checkpoint": best_path,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "history": history,
                "device": str(device),
            },
            indent=2,
        )
    )
    print(f"Wrote {metrics_path}")
    print(f"Done. Live Aura loads this via CONFIG.emotion_checkpoint:\n  {best_path}")


if __name__ == "__main__":
    main()
