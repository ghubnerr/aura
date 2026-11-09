"""Read stored emotion–video pairs from Aura storage."""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np

from aura.config import CONFIG


class PairsProvider:
    """Filesystem reader for generated ``emotion_dataset/<hash>/`` folders."""

    def __init__(
        self,
        storage_path: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self.storage_path = storage_path or CONFIG.storage_path
        self.dataset_path = os.path.join(
            self.storage_path, "aura_storage", "emotion_dataset"
        )
        self.embedding_dim = embedding_dim or CONFIG.emotion_embed_dim

    def load_pair(self, hash_id: str) -> Tuple[str, str, str, str]:
        """Return paths to face, caption text, embedding, and video."""
        folder_path = os.path.join(self.dataset_path, hash_id)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Directory for hash {hash_id} not found.")

        face_image_path = os.path.join(folder_path, "face.jpg")
        caption_path = os.path.join(folder_path, "caption.txt")
        embedding_path = os.path.join(folder_path, "embedding.npy")
        video_path = os.path.join(folder_path, "video.mp4")

        with open(caption_path, "r", encoding="utf-8") as handle:
            caption = handle.read().strip()

        return face_image_path, caption, embedding_path, video_path

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load a BGR image from disk."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found at path: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to decode image at {image_path}")
        return image

    @staticmethod
    def get_video_frames(video_path: str) -> List[np.ndarray]:
        """Extract all frames from a video file as BGR arrays."""
        capture = cv2.VideoCapture(video_path)
        frames: List[np.ndarray] = []
        try:
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        return frames

    def load_embedding(
        self, embedding_path: str, verify_size: bool = False
    ) -> np.ndarray:
        """Load a ``.npy`` emotion embedding."""
        if not os.path.isfile(embedding_path):
            raise FileNotFoundError(
                f"Embedding file not found at path: {embedding_path}"
            )
        if not embedding_path.endswith(".npy"):
            raise ValueError(
                f"Invalid file type: {embedding_path}. Expected a .npy file."
            )

        embedding = np.load(embedding_path)
        if verify_size and embedding.shape[-1] != self.embedding_dim:
            raise AssertionError(
                f"Expected embedding of size {self.embedding_dim}, "
                f"but got size {embedding.shape[-1]} at {embedding_path}."
            )
        return embedding

    @staticmethod
    def get_first_video_frame(video_path: str) -> np.ndarray:
        """Return the first frame of a video."""
        capture = cv2.VideoCapture(video_path)
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            raise ValueError("Failed to read the first frame from the video.")
        return frame

    def get_all_hash_ids(self) -> List[str]:
        """List all pair hash directories under the dataset path."""
        if not os.path.isdir(self.dataset_path):
            return []
        return [
            name
            for name in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, name))
        ]
