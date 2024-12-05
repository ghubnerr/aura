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

DATASET_PATH = "/disk/onyx-scratch/dullo009-fall2024"

class PairsGenerator:
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
    def _generate_text_description(image: np.ndarray, model: str = "ollama/llama3.2-vision") -> str:
        """
        Generates a text description of the emotion image using GPT-4o Vision or Llama3.2-Vision.
        Wrapped in a default prompt engineered message.
        """

        model_opts = model.split("/")
        if len(model_opts) != 2 or (model_opts[0] != "ollama" and model_opts[0] != "openai"):
            raise ValueError("`model` parameter must be in the format `service/model_name`, i.e. `ollama/llama3.2-vision`.")
        service = model_opts[0]
        model = model_opts[1]

        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()

        user_query = """
            Your task is continue a prompt for generating a beautiful and captivating abstract art piece.
            This piece will be described below and you will continue it's text with regards to the provided image.
            The provided image is a person whose emotion or aura you will analyze and continue the prompt accordingly.
            The artwork should express the person's emotion or aura and ensure the prompt isn't made too much longer.
            Do not include any other text besides the full prompt and your continuation. Here's the prompt:

            Generate an abstract, large-scale digital artwork in a modern gallery space. The piece features flowing, organic forms with intricate textures of clusters, ribbons, and particles. Bold, contrasting colors dominate, creating depth, motion, and balance.
            """

        if service == "ollama":
            return ollama.chat(
                model = model,
                messages=[{
                    'role': 'assistant',
                    'content': user_query,
                    'images': [image_bytes]
                }]
            )['message']['content']

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OPENAI_API_KEY not provided")

        openai.api_key = api_key
        client = openai.OpenAI()
        
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4
        )

        return response.choices[0].message.content
            
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
    provider = DatasetProvider(DATASET_PATH)
    emotion_model = EmotionModel()
    video_generator = OpenSoraT2VideoPipeline()
    pair_generator = PairsGenerator(provider, emotion_model, video_generator, True)
    print("Saving aura dataset pairs...")
    response = pair_generator._generate_text_description(provider.sample(1, True)[0])
    print(f"\n{response}\n")