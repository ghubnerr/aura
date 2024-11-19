from typing import *
from sim_clr import SimCLR
from camera import ProcessingPipeline
from tqdm import tqdm
from .provider import DatasetProvider
import openai
from PIL import Image
import io
import os
from dotenv import load_dotenv
import base64
import numpy as np

TEMP_IMAGE_PATH = "./temp"

class PairsGenerator:
    def __init__(self, dataset_provider: DatasetProvider, simclr_model: SimCLR, processing_pipeline: ProcessingPipeline, video_generator):
        self.dataset_provider = dataset_provider
        self.simclr_model = simclr_model
        self.processing_pipeline = processing_pipeline
        self.video_generator = video_generator

    def get_next_pair(self) -> Tuple:
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
        video = self.video_generator(text_description)

        # Preprocess the image using the processing pipeline
        processed_image = self.processing_pipeline.process(image)

        # Embedding for the processed image using SimCLR
        embedding = self.simclr_model.embed(processed_image)

        return embedding, video
    

    def _generate_text_description(self, image: Image.Image) -> str:
        """
        Generates a text description of the emotion image using GPT-4 Vision.
        Wrapped in a default prompt engineered message.
        """

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise Exception("OPENAI_API_KEY not provided")

        openai.api_key = api_key
        
        try:
            # Convert PIL image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            client = openai.OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this person's emotion in up to three words. Make sure to use only nouns or adjectives."
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
            
            # TODO: Add prompt engineering for the model here
            return f"{model_response}"
        
        except Exception as e:
            print(f"Error generating text description: {e}")
            return "Error generating image description."
            
    def save_pairs(self, pair: Tuple, storage_path: str):
        """
        Saves the generated pairs to a specified storage location.

        Args:
        - pair: A tuple containing the image embedding and video.
        - storage_path: Path to the storage location (local - local:// - or Hugging Face dataset - hf://).
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