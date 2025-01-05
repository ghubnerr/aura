import os
import shutil
from typing import *
import base64

import torch
import numpy as np
import cv2
from aura.cnn import EmotionModel
from tqdm import tqdm
from openai import OpenAI
import ollama
from PIL import Image

from aura.utils import hash_image
from aura.dataset.provider import DatasetProvider
from aura.dataset.t2v_model import *


class PairsGenerator:

    CAPTION_PROMPT = """Your task is to fill in the following prompt which will be used in a video generation model. The video generation should create a video which represents the aura of the following person and the video should be art in the style of Refik Anadol. Keep your response short and concise, focusing on the emotions displayed by the person and their abstract "aura". Think about how the video should look if it were to reflect the person's emotions or reaction, incorporating colors and shapes reflective of these emotions or reactions. Return the provided prompt found below continued by your addition - keep your addition concise, but DO NOT PROVIDE ANY OTHER TEXT. YOUR RESPONSE WILL BE AUTOMATICALLY SUPPLIED TO ANOTHER MODEL."""

    VIDEO_PROMPT = "Generate an abstract, large-scale digital artwork in a modern gallery space. The piece features flowing, organic forms with intricate textures of clusters, ribbons, and particles. Bold, contrasting colors dominate, creating depth, motion, and balance."


    def __init__(self, dataset_provider: DatasetProvider, 
                 emotion_model: EmotionModel,
                 video_generator: OpenSoraT2VideoPipeline|CogVideoXT2VideoPipeline|LatteT2VideoPipeline,
                 **kwargs):
        self.dataset_provider = dataset_provider
        self.emotion_model = emotion_model
        self.video_generator = video_generator
        self.openai_client = OpenAI()

        self._current_image = 0

        self.dataset_path = os.path.join(os.environ.get("STORAGE_PATH"), "aura_storage", "emotion_dataset")
        os.makedirs(self.dataset_path, exist_ok = True)

    def _generate_caption(self, image: np.ndarray, use_openai: bool = False) -> str:
        """
        Generates a text description of the emotion image using Ollama, by default, with the Llama3.2-Vision model.
        Wrapped in a default prompt engineered message.
        """
        
        if isinstance(image, Image.Image):
            print("Converting to continguous array")
            image = np.ascontiguousarray(image, dtype=np.uint8)

        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()

        if use_openai:
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            messages = [
                {
                    "role": "developer",
                    "content": [
                        {"type": "text", "text": PairsGenerator.CAPTION_PROMPT},
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PairsGenerator.VIDEO_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ]

            response = self.openai_client.chat.completions.create(
                model= "gpt-4o-mini",
                messages=messages,
            )

            return response.choices[0].message.content

        return ollama.chat(
            model = "llama3.2-vision",
            messages=[{
                'role': 'assistant',
                'content': PairsGenerator.DESCRIPTION_PROMPT,
                'images': [image_bytes]
            }, {
                'role': 'user',
                'content': PairsGenerator.VIDEO_PROMPT,
            }],
            options = {
                'temperature': 0
            }
        )['message']['content']
            
    def _save_to_hf(self, image: np.ndarray, caption: str, embedding: torch.Tensor, video_path: str):
        """
        Saves the generated pairs to a specified storage location.

        Args:
        - pair: A tuple containing the image embedding and video.
        - storage_path: Path to the storage location (local - local:// \
                        - or Hugging Face dataset - hf://).
        """
        import huggingface_hub
        huggingface_hub.login()

        from dataset import Dataset
        
        data = {
            "image": image,
            "description": caption,
            "embedding": [embedding.detach().numpy().tolist()],  
            # TODO: Read Video from `video_path` and save to huggingface
            "video": [video], 
        }
        dataset = Dataset.from_dict(data)

        # Handle duplicate avoidance using unique identifiers
        existing_dataset = huggingface_hub.hf_hub_download(dataset_name, "pairs")
        if existing_dataset:
            _ = Dataset.load_from_disk(existing_dataset)

            # Check for duplicates based on embedding and video hash
            new_data = dataset.filter(lambda x: hash(x["image"]) != hash(image))
            if len(new_data) < len(dataset):
                print(f"Duplicate detected, skipping save for pair: {id}")
                return

        dataset.push_to_hub(dataset_name)
        print(f"Pair saved to Hugging Face dataset: {dataset_name}")

    def save_pairs(self, count: int, huggingface: bool = False):
        for image, caption, embedding in self.generate_pairs(count):

            img_hash = hash_image(image)
            video_path = os.path.join(self.dataset_path, img_hash, "video.mp4")
            self.video_generator(caption, path = video_path, aspect_ratio = "1:1", num_frames = "16s", resolution = "480p")

            if huggingface:
                self._save_to_hf(image, caption, embedding)
                continue

            image_path = os.path.join(self.dataset_path, img_hash, "face.jpg")
            cv2.imwrite(image_path, image)

            description_path = os.path.join(self.dataset_path, img_hash, "caption.txt")
            with open(description_path, "w", encoding = "utf-8") as file:
                file.write(caption)

            embedding_path = os.path.join(self.dataset_path, img_hash, "embedding.npy")
            with open(embedding_path, "wb") as f:
                np.save(f, embedding.detach().numpy())


    def generate_pairs(self, count: int):
        """
        Generates and saves a specified number of pairs to a given path.
        """
        for _ in tqdm(range(count), desc="Generating pairs"):
            image, _, _ = self.dataset_provider.sample(self._current_image)
            self._current_image += 1
            if image is None:
                raise StopIteration("No more images available in the dataset.")

            img_hash = hash_image(image)
            folder_path = os.path.join(self.dataset_path, img_hash)

            try:
                os.makedirs(folder_path)
            except:
                print("Image already processed...overwritting")
                shutil.rmtree(folder_path)
                os.makedirs(folder_path)
            
            
            caption = self._generate_caption(image, True)
            embedding = self.emotion_model.embed(image)
            yield image, caption, embedding


if __name__ == "__main__":
    provider = DatasetProvider()
    emotion_model = EmotionModel()
    video_generator = OpenSoraT2VideoPipeline(num_gpus = 4)
    pair_generator = PairsGenerator(provider, emotion_model, video_generator)

    path = os.path.join(os.environ.get("STORAGE_PATH"), "aura_storage", "emotion_dataset")
    os.makedirs(path, exist_ok = True)
    pair_generator.save_pairs(2_500)
