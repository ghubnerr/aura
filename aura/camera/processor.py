"""Face detection and preprocessing for the emotion encoder."""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np


class FaceNotFoundException(Exception):
    """Raised when no face is detected in an image."""

    def __init__(self, message: str = "Face not found in the image") -> None:
        super().__init__(message)


class ProcessingPipeline:
    """Detect the largest face and prepare a CNN-ready tensor."""

    def __init__(self, log_path: str = "./logs", verbose: int = 0) -> None:
        """Create a face-processing pipeline.

        Args:
            log_path: Directory for optional debug images / logs.
            verbose: 0 = quiet, 1 = save images, 2 = save images + timing logs.

        Raises:
            ValueError: If verbose is outside [0, 2].
        """
        if not 0 <= verbose <= 2:
            raise ValueError("ProcessingPipeline verbosity must be between 0 and 2.")

        self.log_path = log_path
        self.verbose = verbose
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.current_log_dir: Optional[str] = None

        if self.verbose > 0:
            self._setup_logging()

    def _setup_logging(self) -> None:
        """Create a timestamped log directory and optional file logger."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_dir = os.path.join(self.log_path, timestamp)
        os.makedirs(self.current_log_dir, exist_ok=True)

        if self.verbose == 2:
            logging.basicConfig(
                filename=os.path.join(self.log_path, f"{timestamp}.log"),
                level=logging.INFO,
                format="%(asctime)s - %(message)s",
            )

    def _save_to_log_path(self, image: np.ndarray, filename: str) -> None:
        """Save an image when verbose logging is enabled."""
        if self.verbose <= 0:
            return
        if self.current_log_dir is None:
            self._setup_logging()
        assert self.current_log_dir is not None
        cv2.imwrite(os.path.join(self.current_log_dir, filename), image)

    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR images to grayscale; leave grayscale unchanged."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def get_bounding_box(
        self, image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return (x, y, w, h) for the largest detected face, or None."""
        start = time.time()
        gray = self._convert_to_gray(image)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if self.verbose == 2:
            logging.info("Face detection took %.2f seconds", time.time() - start)

        if len(faces) == 0:
            return None
        return sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[0]

    def annotate_face(self, image: np.ndarray) -> np.ndarray:
        """Draw a rectangle around the largest face and optionally log it."""
        bbox = self.get_bounding_box(image)
        if bbox is None:
            return image

        x, y, w, h = bbox
        annotated = image.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
        self._save_to_log_path(annotated, "bbox.jpg")
        return annotated

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Crop, resize, and normalize the largest face.

        Args:
            image: BGR image from OpenCV.

        Returns:
            Float32 array shaped (3, 224, 224) with values in [0, 1].

        Raises:
            FaceNotFoundException: If no face is found.
        """
        start = time.time()
        bbox = self.get_bounding_box(image)
        if bbox is None:
            raise FaceNotFoundException()

        x, y, w, h = bbox
        face = image[y : y + h, x : x + w]
        face = cv2.resize(face, (224, 224))
        face = face.astype(np.float32) / 255.0
        face = np.transpose(face, (2, 0, 1))

        if self.verbose > 0:
            preview = (face * 255).transpose(1, 2, 0).astype(np.uint8)
            self._save_to_log_path(preview, "processed.jpg")
        if self.verbose == 2:
            logging.info("Image processing took %.2f seconds", time.time() - start)

        return face


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and annotate faces in an image."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument(
        "--log_path",
        type=str,
        default="./logs",
        help="Directory for logs and outputs",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        raise SystemExit(f"Error: The file '{args.image_path}' does not exist.")

    pipeline = ProcessingPipeline(log_path=args.log_path, verbose=2)
    image = cv2.imread(args.image_path)
    if image is None:
        raise SystemExit(
            f"Error: Unable to read the image at '{args.image_path}'."
        )

    try:
        annotated = pipeline.annotate_face(image)
        output_path = os.path.join(
            pipeline.current_log_dir or args.log_path, "bbox.jpg"
        )
        cv2.imwrite(output_path, annotated)
        print(f"Annotated image saved at '{output_path}'")
    except FaceNotFoundException as exc:
        print(exc)


if __name__ == "__main__":
    _main()
