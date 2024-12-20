import base64
import os
from typing import *

import numpy as np
import cv2
from aura.embed import EmotionModel
from aura.camera import ProcessingPipeline
from tqdm import tqdm
import openai
from dotenv import load_dotenv
import ollama

from .provider import DatasetProvider
from .t2v_model import *

class PairsGenerator:

    DESCRIPTION_PROMPT = """You are a modern art prompt specialist, trained to create prompts for abstract art using diffusion models. Your task is to generate a detailed prompt for an abstract art piece that features flowing ribbons, dynamic circles, particles, and an overall sense of motion, all highlighted by vivid, shifting colors. The style should evoke the signature work of Anadol Refik, blending fluidity, motion, and vibrant hues.
    The starting point for this artwork is an image representing a human reaction to the art. This emotional or psychological response should be mirrored in the visual elements of the artwork. The colors, shapes, and movement should reflect the nature of this reaction. For example, a sense of calm could be represented by gentle, slow-moving ribbons, while excitement might bring faster, more erratic motion with bursts of vibrant colors.
    Given an image of a person and an initial starting prompt, create a better and concise prompt which reflects the person's emotion, based on the image.
    """
    VIDEO_PROMPT = "Generate an abstract, large-scale digital artwork. The piece features flowing, organic forms with intricate textures of clusters, ribbons, and particles. Bold, contrasting colors dominate, creating depth, motion, and balance."


    def __init__(self, dataset_provider: DatasetProvider, 
                 emotion_model: EmotionModel,
                 video_generator: OpenSoraT2VideoPipeline|CogVideoXT2VideoPipeline|LatteT2VideoPipeline,
                 test: bool,
                 **kwargs):
        self.dataset_provider = dataset_provider
        self.emotion_model = emotion_model
        self.video_generator = video_generator
        self.test = test

        self._current_image = 0

    def get_next_pair(self, **kwargs) -> Tuple:
        """
        Generates the next image embedding and video pair.
        """

        image = self.dataset_provider.sample(self._current_image, test = self.test)
        self.current_image += 1
        if image is None:
            raise StopIteration("No more images available in the dataset.")

        text_description = self._generate_text_description(image)
        video = self.video_generator(text_description, **kwargs)
        embedding = self.emotion_model.embed(image)

        return image, embedding, video, text_description
    
    @staticmethod
    def _generate_text_description(image: np.ndarray, model: str = "llama3.2-vision") -> str:
        """
        Generates a text description of the emotion image using Ollama, by default, with the Llama3.2-Vision model.
        Wrapped in a default prompt engineered message.
        """

        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()

        return ollama.chat(
            model = model,
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
            
    def save_pairs(self, pair: Tuple, storage_path: str):
        """
        Saves the generated pairs to a specified storage location.

        Args:
        - pair: A tuple containing the image embedding and video.
        - storage_path: Path to the storage location (local - local:// \
                        - or Hugging Face dataset - hf://).
        """

        image, embedding, video, text_description = pair
        id = hash(image)

        if storage_path.startswith("local://"):
            local_path = storage_path.replace("local://", "")
            os.makedirs(local_path, exist_ok=True)  

            folder = os.path.join(local_path, id)
            os.makedirs(folder)

            image_path = os.path.join(folder, f"image_{id}.jpg")
            cv2.imwrite(image_path, image)

            description_path = os.path.join(folder, f"description_{id}.txt")
            with open(description_path, "w") as file:
                file.write(text_description)

            embedding_path = os.path.join(folder, f"embedding_{id}.npy")
            with open(embedding_path, "wb") as f:
                np.save(f, embedding)

            video_path = os.path.join(local_path, f"video_{id}.mp4")
            with open(video_path, "wb") as f:
                f.write(video)
            
            print(f"Pair saved locally: {embedding_path}, {video_path}")

        elif storage_path.startswith("hf://"):
            import huggingface_hub

            dataset_name = storage_path.replace("hf://", "")
            huggingface_hub.login()

            from dataset import Dataset
            
            data = {
                "image": image,
                "description": text_description,
                "embedding": [embedding.tolist()],  
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

    def generate_and_save_pairs(self, count: int, storage_path: str):
        """
        Generates and saves a specified number of pairs to a given path.
        """
        for _ in tqdm(range(count), desc="Generating pairs"):
            pair = self.get_next_pair()
            self.save_pairs(pair, storage_path)

    def generate_and_yield_pairs(self, count: int) -> Iterator[Tuple]:
        """
        Generates and yields a specified number of pairs.      
        """
        for _ in tqdm(range(count), desc="Generating pairs"):
            yield self.get_next_pair()

if __name__ == "__main__":
    provider = DatasetProvider()
    emotion_model = EmotionModel()
    video_generator = OpenSoraT2VideoPipeline()
    pair_generator = PairsGenerator(provider, emotion_model, video_generator, True)

    print("Generating prompt...")
    response = pair_generator._generate_text_description(provider.sample(8, True)[0])
    print(f"\n{response}\n")
    print("Generating video...")
    video_generator(response, f'{os.environ.get("STORAGE_PATH")}/aura_storage/ollama_example.mp4')
