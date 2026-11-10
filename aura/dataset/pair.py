"""Generate (face, caption, emotion embedding, abstract video) training pairs."""

from __future__ import annotations

import base64
import os
import shutil
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np
import ollama
import torch
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from aura.cnn import EmotionModel
from aura.config import CONFIG
from aura.dataset.dataset_provider import DatasetProvider
from aura.dataset.t2v_model import (
    CogVideoXT2VideoPipeline,
    LatteT2VideoPipeline,
    OpenSoraT2VideoPipeline,
    WanT2VideoPipeline,
    build_t2v_pipeline,
)
from aura.utils import hash_image

VideoGenerator = Union[
    OpenSoraT2VideoPipeline,
    CogVideoXT2VideoPipeline,
    LatteT2VideoPipeline,
    WanT2VideoPipeline,
]


class PairsGenerator:
    """Create emotion-aligned synthetic abstract-art video pairs."""

    CAPTION_PROMPT = (
        "Your task is to fill in the following prompt which will be used in a "
        "video generation model. The video generation should create a video which "
        "represents the aura of the following person and the video should be art "
        "in the style of Refik Anadol. Keep your response short and concise, "
        "focusing on the emotions displayed by the person and their abstract "
        '"aura". Think about how the video should look if it were to reflect the '
        "person's emotions or reaction, incorporating colors and shapes reflective "
        "of these emotions or reactions. Return the provided prompt found below "
        "continued by your addition - keep your addition concise, but DO NOT "
        "PROVIDE ANY OTHER TEXT. YOUR RESPONSE WILL BE AUTOMATICALLY SUPPLIED TO "
        "ANOTHER MODEL."
    )

    VIDEO_PROMPT = (
        "Generate an abstract, large-scale digital artwork in a modern gallery "
        "space. The piece features flowing, organic forms with intricate textures "
        "of clusters, ribbons, and particles. Bold, contrasting colors dominate, "
        "creating depth, motion, and balance."
    )

    def __init__(
        self,
        dataset_provider: DatasetProvider,
        emotion_model: EmotionModel,
        video_generator: VideoGenerator,
    ) -> None:
        self.dataset_provider = dataset_provider
        self.emotion_model = emotion_model
        self.video_generator = video_generator
        self.openai_client = OpenAI()
        self._current_image = 0
        self.dataset_path = CONFIG.emotion_dataset_dir
        os.makedirs(self.dataset_path, exist_ok=True)

    def _generate_caption(
        self, image: np.ndarray, use_openai: bool = True
    ) -> str:
        """Caption a face image with an abstract-art video prompt."""
        if isinstance(image, Image.Image):
            image = np.ascontiguousarray(image, dtype=np.uint8)

        _ok, buffer = cv2.imencode(".jpg", image)
        image_bytes = buffer.tobytes()

        if use_openai:
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            messages = [
                {
                    "role": "developer",
                    "content": [
                        {"type": "text", "text": self.CAPTION_PROMPT},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.VIDEO_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                },
            ]
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            return response.choices[0].message.content or self.VIDEO_PROMPT

        response = ollama.chat(
            model="llama3.2-vision",
            messages=[
                {
                    "role": "assistant",
                    "content": self.CAPTION_PROMPT,
                    "images": [image_bytes],
                },
                {
                    "role": "user",
                    "content": self.VIDEO_PROMPT,
                },
            ],
            options={"temperature": 0},
        )
        return response["message"]["content"]

    def _save_to_hf(
        self,
        image: np.ndarray,
        caption: str,
        embedding: np.ndarray,
        video_path: str,
    ) -> None:
        """Upload a pair to Hugging Face Hub (not implemented yet)."""
        del image, caption, embedding, video_path
        raise NotImplementedError("Hugging Face pair upload is not implemented yet.")

    def save_pairs(self, count: int, huggingface: bool = False) -> None:
        """Generate and persist ``count`` face/video pairs."""
        for image, caption, embedding in self.generate_pairs(count):
            img_hash = hash_image(image)
            folder = os.path.join(self.dataset_path, img_hash)
            os.makedirs(folder, exist_ok=True)
            video_path = os.path.join(folder, "video.mp4")
            self.video_generator(caption, path=video_path)

            if huggingface:
                self._save_to_hf(image, caption, embedding, video_path)
                continue

            cv2.imwrite(os.path.join(folder, "face.jpg"), image)
            with open(
                os.path.join(folder, "caption.txt"), "w", encoding="utf-8"
            ) as handle:
                handle.write(caption)
            np.save(os.path.join(folder, "embedding.npy"), embedding)

    def generate_pairs(
        self, count: int
    ) -> Iterator[Tuple[np.ndarray, str, np.ndarray]]:
        """Yield ``(face_bgr, caption, emotion_embedding)`` tuples."""
        for _ in tqdm(range(count), desc="Generating pairs"):
            image, _label, _emotion = self.dataset_provider.sample(self._current_image)
            self._current_image += 1

            img_hash = hash_image(image)
            folder_path = os.path.join(self.dataset_path, img_hash)
            if os.path.exists(folder_path):
                print(f"Image {img_hash} already processed; overwriting.")
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)

            caption = self._generate_caption(image, use_openai=True)
            embedding = self.emotion_model.embed(image, use_representation=True)
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            yield image, caption, np.asarray(embedding)


if __name__ == "__main__":
    provider = DatasetProvider()
    emotion_model = EmotionModel(pretrained=True)
    emotion_model.to(emotion_model.device)
    video_generator = build_t2v_pipeline("wan")
    pair_generator = PairsGenerator(provider, emotion_model, video_generator)
    pair_generator.save_pairs(100)
