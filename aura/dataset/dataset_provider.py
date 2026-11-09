"""AffectNet dataset loader used to train the emotion CNN and build pairs."""

from __future__ import annotations

import os
import random
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np

from aura.config import CONFIG

CLASS_CUTOFF = 2000
EmotionSample = Tuple[np.ndarray, int, str]


class DatasetProvider:
    """Download / cache AffectNet and yield labeled face crops."""

    def __init__(
        self,
        target_size: Union[None, Tuple[int, int]] = None,
        augment: bool = False,
        split: Optional[float] = None,
    ) -> None:
        self.target_size = target_size
        self.emotion_labels: Dict[str, int] = {
            "happy": 0,
            "sad": 1,
            "anger": 2,
            "disgust": 3,
            "fear": 4,
            "neutral": 5,
            "surprise": 6,
            "contempt": 7,
        }

        import kagglehub  # Lazy: only required when constructing a DatasetProvider.

        original_dir = os.getcwd()
        try:
            os.makedirs(CONFIG.emotion_dataset_dir, exist_ok=True)
            os.chdir(CONFIG.emotion_dataset_dir)
            dataset_path = kagglehub.dataset_download(
                "noamsegal/affectnet-training-data"
            )
        except OSError as exc:
            print(
                f"Could not access aura storage ({CONFIG.storage_path}): {exc}. "
                "Downloading in the process directory."
            )
            dataset_path = kagglehub.dataset_download(
                "noamsegal/affectnet-training-data"
            )
        finally:
            os.chdir(original_dir)

        self.split = split
        self.dataset = self._collect_files(dataset_path, augment)
        self.train: List[EmotionSample] = self.dataset
        self.test: List[EmotionSample] = []
        if split is not None:
            train_size = int(len(self.dataset) * split)
            self.train = self.dataset[:train_size]
            self.test = self.dataset[train_size:]

    def sample(
        self,
        index: int,
        source: Optional[Literal["train", "test"]] = None,
    ) -> EmotionSample:
        """Return ``(image, label_id, emotion_name)`` for ``index``."""
        dataset = self._get_dataset(source)
        if index >= len(dataset):
            raise ValueError(
                f"Index {index} is larger than dataset length {len(dataset)}"
            )
        return dataset[index]

    def get_next_image_batch(
        self,
        batch_size: int,
        source: Optional[Literal["train", "test"]] = None,
    ) -> Iterator[List[EmotionSample]]:
        """Yield successive batches from the requested split."""
        if source == "test" and self.split is None:
            raise ValueError(
                "Batch requested from test split, but `split` was not set."
            )

        dataset = self._get_dataset(source)
        if batch_size > len(dataset):
            raise ValueError("Can't get batch size larger than dataset")

        for start in range(0, len(dataset), batch_size):
            batch = [
                self.sample(idx, source)
                for idx in range(start, min(start + batch_size, len(dataset)))
            ]
            yield batch

    def _get_dataset(
        self, source: Optional[Literal["train", "test"]] = None
    ) -> List[EmotionSample]:
        if source == "train":
            return self.train
        if source == "test":
            return self.test
        return self.dataset

    def label_distr(self, test: bool = False) -> None:
        """Print emotion class distribution for train or test."""
        dataset = self.test if test else self.train
        counter = Counter(emotion for _, _, emotion in dataset)
        for emotion, count in counter.items():
            print(f"{emotion}: {(count / len(dataset)) * 100:.2f}%")

    def _get_emotion(self, path: str) -> str:
        dirs = path.replace("\\", "/").split("/")
        if len(dirs) < 2:
            raise ValueError(f"Path {path} is not in a parseable emotion format")
        return dirs[-2]

    def _collect_files(
        self, path: str, augment: bool = False
    ) -> List[EmotionSample]:
        dataset: List[EmotionSample] = []
        label_counter: Dict[str, int] = defaultdict(int)

        for dirpath, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                img_path = os.path.join(dirpath, filename)
                if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                emotion = self._get_emotion(img_path)
                if emotion not in self.emotion_labels:
                    continue
                if label_counter[emotion] >= CLASS_CUTOFF:
                    continue

                label = self.emotion_labels[emotion]
                img = cv2.imread(img_path)
                if img is None:
                    continue

                variants = [img]
                if augment:
                    variants.extend(
                        [
                            self._random_rotation(img),
                            self._flip(img),
                            self._random_brightness(img),
                        ]
                    )

                for variant in variants:
                    dataset.append((variant, label, emotion))
                label_counter[emotion] += len(variants)

        random.shuffle(dataset)
        return dataset

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if self.target_size is None:
            raise ValueError("Called `_resize_image` without set target_size.")
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array.")
        if image.ndim not in (2, 3):
            raise ValueError("Input image must be grayscale or BGR.")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("Input image has invalid dimensions.")

        scale = min(self.target_size[0] / height, self.target_size[1] / width)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros(
            (self.target_size[0], self.target_size[1], 3), dtype=np.uint8
        )
        top = (self.target_size[0] - new_h) // 2
        left = (self.target_size[1] - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return np.ascontiguousarray(canvas, dtype=np.uint8)

    def _random_rotation(self, image: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(-30, 30)
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
        return cv2.warpAffine(image, matrix, (width, height))

    def _flip(self, image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 1)

    def _random_brightness(self, image: np.ndarray) -> np.ndarray:
        factor = np.random.uniform(0.5, 1.5)
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)

    def _random_zoom(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        zoom = np.random.uniform(0.3, 2.0)
        new_h, new_w = int(height * zoom), int(width * zoom)
        resized = cv2.resize(image, (new_w, new_h))
        if new_h < height or new_w < width:
            top = (height - new_h) // 2
            left = (width - new_w) // 2
            cropped = resized[top : top + new_h, left : left + new_w]
        else:
            cropped = resized[:height, :width]
        return self._resize_image(cropped)

    def _set_black_background(
        self, image: np.ndarray, threshold: int = 20
    ) -> np.ndarray:
        image = np.ascontiguousarray(image, dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        image[mask == 255] = [0, 0, 0]
        return image
