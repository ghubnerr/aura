#!/usr/bin/env python3
"""Verify the emotion CNN checkpoint exists and loads."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from aura.config import CONFIG


def main() -> int:
    path = Path(CONFIG.emotion_checkpoint)
    metrics = Path(CONFIG.aura_storage) / "emotion_train_metrics.json"
    print(f"STORAGE_PATH={CONFIG.storage_path}")
    print(f"checkpoint={path}")
    print(f"exists={path.is_file()}")
    if path.is_file():
        print(f"size_bytes={path.stat().st_size}")
    if metrics.is_file():
        data = json.loads(metrics.read_text())
        print(f"metrics={json.dumps(data, indent=2)}")
    else:
        print("metrics=missing")

    if not path.is_file():
        print("FAIL: no checkpoint — live path will use demo heuristic")
        return 1

    import torch
    from aura.cnn import EmotionModel

    model = EmotionModel(pretrained=False)
    model.load(str(path))
    print(f"device={model.device}")
    print("OK: checkpoint loads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
