from typing import *
from aura.sim_clr import SimCLR
from aura.camera import ProcessingPipeline
from tqdm import tqdm
from .provider import DatasetProvider
import openai
from PIL import Image
import io
import os
from dotenv import load_dotenv
from .t2v_model import *
import base64
import json
import numpy as np

TEMP_IMAGE_PATH = "./temp"

class PairsGenerator:
    def __init__(self, dataset_provider: DatasetProvider, 
                 simclr_model: SimCLR, processing_pipeline: ProcessingPipeline, 
                 video_generator: OpenSoraT2VideoPipeline|CogVideoXT2VideoPipeline,
                 **kwargs):
        self.dataset_provider = dataset_provider
        self.simclr_model = simclr_model
        self.processing_pipeline = processing_pipeline
        self.video_generator = video_generator(**kwargs)

    def get_next_pair(self, **kwargs) -> Tuple:
        """
        Generates the next image embedding and video pair.
        """
        # Get an image from the dataset provider
        image = self.dataset_provider.get_image()
        if image is None:
            raise StopIteration("No more images available in the dataset.")

        # Generate a text description of the image using GPT-4
        text_description = self._generate_text_description(image)

        # Generate a video from the text description
        video = self.video_generator(text_description, **kwargs)

        # Preprocess the image using the processing pipeline
        processed_image = self.processing_pipeline.process(image)

        # Embedding for the processed image using SimCLR
        embedding = self.simclr_model.embed(processed_image)

        return embedding, video
    
    @staticmethod
    def _generate_text_description(image: Image.Image, model: str = "ollama/bakllava") -> str:
        """
        Generates a text description of the emotion image using GPT-4 Vision.
        Wrapped in a default prompt engineered message.
        """
        # Convert PIL image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        user_query = """
            Look at this image carefully and analyze the person's facial expression.
            What emotion is being displayed?
            Respond with 1-3 emotion-related words only (e.g., 'happy', 'sad', 'angry', etc.).
            Do not include percentages or technical details.
            """

        if model.startswith("ollama"):
            import requests
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model.split("/")[1],
                    "prompt": user_query,
                    "images": [base64_image],
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Error from Ollama API: {response.text}")
            
            responses = [json.loads(line) for line in response.text.strip().split('\n')]        
            
            final_response = ""
            for resp in responses:
                if "response" in resp:
                    final_response += resp["response"]
                if "error" in resp:
                    print(f"Error in response: {resp['error']}")
            
            model_response = final_response.strip()
            
            if not model_response:
                raise Exception("Empty response from Ollama")
                        
            return model_response

        else: # OPENAI_API_KEY-based model i.e. "gpt-4-vision-preview"
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise Exception("OPENAI_API_KEY not provided")

            openai.api_key = api_key

            client = openai.OpenAI()
            
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

            model_response = response.choices[0].message.content
                    
        return f"Abstract art with swirls, curves, color, light, synesthesia, and shapes \
                 which represent the following emotion or idea: {model_response}"
            
    def save_pairs(self, pair: Tuple, storage_path: str):
        """
        Saves the generated pairs to a specified storage location.

        Args:
        - pair: A tuple containing the image embedding and video.
        - storage_path: Path to the storage location (local - local:// \
                        - or Hugging Face dataset - hf://).
        """

        embedding, video = pair

        if storage_path.startswith("local://"):
            local_path = storage_path.replace("local://", "")
            os.makedirs(local_path, exist_ok=True)  

            file_id = f"{hash(tuple(embedding))}_{len(video)}"
            embedding_path = os.path.join(local_path, f"embedding_{file_id}.npy")
            video_path = os.path.join(local_path, f"video_{file_id}.mp4")

            with open(embedding_path, "wb") as f:
                np.save(f, embedding)

            with open(video_path, "wb") as f:
                f.write(video)

            print(f"Pair saved locally: {embedding_path}, {video_path}")

        elif storage_path.startswith("hf://"):
            import huggingface_hub

            dataset_name = storage_path.replace("hf://", "")
            huggingface_hub.login()

            from datasets import Dataset
            
            data = {
                "embedding": [embedding.tolist()],  
                "video": [video], 
            }
            dataset = Dataset.from_dict(data)

            # Handle duplicate avoidance using unique identifiers
            existing_dataset = huggingface_hub.hf_hub_download(dataset_name, "pairs")
            if existing_dataset:
                _ = Dataset.load_from_disk(existing_dataset)

                # Check for duplicates based on embedding and video hash
                new_data = dataset.filter(lambda x: hash(tuple(x["embedding"])) != hash(tuple(embedding)))
                if len(new_data) < len(dataset):
                    print(f"Duplicate detected, skipping save for pair: {file_id}")
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

