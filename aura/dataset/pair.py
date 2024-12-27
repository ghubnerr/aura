import os
from typing import *
import base64
from io import BytesIO

import numpy as np
import cv2
from aura.cnn import EmotionModel
from aura.camera import ProcessingPipeline
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import ollama
from PIL import Image


from aura.utils import hash_image
from .provider import DatasetProvider
from .t2v_model import *

class PairsGenerator:

    CAPTION_PROMPT = """Your task is to fill in the following prompt which will be used in a video generation model. The video generation should create a video which represents the aura of the following person and the video should be art in the style of Refik Anadol. Keep your response short and concise, focusing on the emotions displayed by the person and their abstract "aura". Think about how the video should look if it were to reflect the person's emotions or reaction. Provide the provided prompt found below continued by your addition - keep your addition concise, but DO NOT PROVIDE ANY OTHER TEXT. YOUR RESPONSE WILL BE AUTOMATICALLY SUPPLIED TO ANOTHER MODEL.
    Generate an abstract, large-scale digital artwork in a modern gallery space. The piece features flowing, organic forms with intricate textures of clusters, ribbons, and particles. Bold, contrasting colors dominate, creating depth, motion, and balance."""

    VIDEO_PROMPT = "Generate an abstract, large-scale digital artwork. The piece features flowing, organic forms with intricate textures of clusters, ribbons, and particles. Bold, contrasting colors dominate, creating depth, motion, and balance."


    def __init__(self, dataset_provider: DatasetProvider, 
                 emotion_model: EmotionModel,
                 video_generator: OpenSoraT2VideoPipeline|CogVideoXT2VideoPipeline|LatteT2VideoPipeline,
                 **kwargs):
        self.dataset_provider = dataset_provider
        self.emotion_model = emotion_model
        self.video_generator = video_generator
        self.openai_client = OpenAI()

        self._current_image = 0

    def get_next_pair(self, **kwargs) -> Tuple:
        """
        Generates the next image embedding and video pair.
        """

        image = self.dataset_provider.sample(self._current_image)
        self.current_image += 1
        if image is None:
            raise StopIteration("No more images available in the dataset.")

        text_description = self._generate_text_description(image, True)
        video = self.video_generator(text_description, **kwargs)
        embedding = self.emotion_model.embed(image)

        return image, embedding, video, text_description
    
    def _generate_text_description(self, image: np.ndarray, use_openai: bool = False) -> str:
        """
        Generates a text description of the emotion image using Ollama, by default, with the Llama3.2-Vision model.
        Wrapped in a default prompt engineered message.
        """
        
        if isinstance(image, Image.Image):
            image = np.array(image)

        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()

        if use_openai:
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PairsGenerator.CAPTION_PROMPT},
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
            
    def save_pairs(self, pair: Tuple, storage_path: str):
        """
        Saves the generated pairs to a specified storage location.

        Args:
        - pair: A tuple containing the image embedding and video.
        - storage_path: Path to the storage location (local - local:// \
                        - or Hugging Face dataset - hf://).
        """

        image, embedding, video, text_description = pair
        id = hash_image(image)

        if storage_path.startswith("local://"):
            local_path = storage_path.replace("local://", "")
            os.makedirs(local_path, exist_ok=True)  

            folder = os.path.join(local_path, id)
            os.makedirs(folder)

            image_path = os.path.join(folder, "face.jpg")
            cv2.imwrite(image_path, image)

            description_path = os.path.join(folder, "caption.txt")
            with open(description_path, "w") as file:
                file.write(text_description)

            embedding_path = os.path.join(folder, "embedding.npy")
            with open(embedding_path, "wb") as f:
                np.save(f, embedding)

            video_path = os.path.join(local_path, "video.mp4")
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

    pair_generator.generate_and_save_pairs(5, 'local://{os.environ.get("STORAGE_PATH")}/aura_storage/ollama_example.mp4')
