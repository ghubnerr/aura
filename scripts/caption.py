"""Batch-caption AffectNet faces via the OpenAI Batch API."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from openai import OpenAI

from aura.dataset import DatasetProvider

SYS_PROMPT = (
    "Imagine a short description of an art piece that would represent this person "
    "as a scene, with objects, weather, and motion. Respond with only the "
    "description inside quotation marks and ignore the fact that the image is in "
    "black and white. Have the description focus on the physical appearance of "
    "the art piece, letting us imagine what it looks like solely based on the "
    "description"
)

client = OpenAI()


def image_to_jpeg_base64(image: np.ndarray) -> str:
    """Encode a BGR/RGB ndarray as a JPEG data-URL payload (base64 only)."""
    if image.ndim == 3 and image.shape[2] == 3:
        # Assume OpenCV BGR; encode as JPEG.
        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Failed to JPEG-encode image")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")
    return base64.b64encode(image.tobytes()).decode("utf-8")


def generate_files(samples: Sequence[Tuple[np.ndarray, int, str]]) -> None:
    batch_size = 5000
    tasks: List[Dict[str, Any]] = []
    metadata: List[Dict[str, Any]] = []

    for index, (image, _label, emotion) in enumerate(samples):
        image_b64 = image_to_jpeg_base64(image)
        tasks.append(
            {
                "custom_id": f"task-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.7,
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": SYS_PROMPT}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    },
                                }
                            ],
                        },
                    ],
                },
            }
        )
        metadata.append({"id": index, "emotion": emotion})

    os.makedirs("output/tasks", exist_ok=True)
    chunk = 1
    for start in range(0, len(tasks), batch_size):
        path = f"output/tasks/aura_caption_tasks_{chunk}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for obj in tasks[start : start + batch_size]:
                handle.write(json.dumps(obj) + "\n")
        chunk += 1

    with open("output/metadata.jsonl", "w", encoding="utf-8") as handle:
        for obj in metadata:
            handle.write(json.dumps(obj) + "\n")
    print(f"Wrote {len(tasks)} caption tasks.")


def submit_batches() -> None:
    batches = 0
    for name in os.listdir("output/tasks"):
        task_path = os.path.join("output/tasks", name)
        if not os.path.isfile(task_path):
            continue
        batches += 1
        batch_file = client.files.create(file=open(task_path, "rb"), purpose="batch")
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(batch_job.id)
    print(f"{batches} batches submitted.")


def main() -> None:
    mode = input(
        "Are you generating task/metadata json files or submitting a batch job "
        "(generate / batch / status)? "
    ).strip()
    if mode == "generate":
        provider = DatasetProvider()
        generate_files(provider.dataset)
    elif mode == "batch":
        submit_batches()
    elif mode == "status":
        for batch_job in list(client.batches.list())[::-1]:
            print(f"\n{batch_job}\n")
    else:
        raise SystemExit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
