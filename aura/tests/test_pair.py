"""PairsGenerator caption smoke tests (skipped without Ollama)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from aura.dataset import PairsGenerator


@pytest.fixture
def random_image() -> np.ndarray:
    image = Image.new("RGB", (100, 100), "blue")
    return np.array(image)[:, :, ::-1]  # BGR like OpenCV


@pytest.fixture
def ensure_ollama_running():
    try:
        subprocess.run(
            ["ollama", "list"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        pytest.skip("Ollama CLI is not installed.")
    except subprocess.CalledProcessError:
        pytest.skip("Ollama is not running.")


def test_generate_caption_ollama(random_image, ensure_ollama_running):
    generator = PairsGenerator(
        dataset_provider=MagicMock(),
        emotion_model=MagicMock(),
        video_generator=MagicMock(),
    )
    description = generator._generate_caption(random_image, use_openai=False)
    assert isinstance(description, str)
    assert len(description) > 0
