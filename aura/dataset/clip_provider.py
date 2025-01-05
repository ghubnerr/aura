import os
import numpy as np
import cv2
from typing import Tuple, List, Union

class PairsProvider:
    def __init__(self, storage_path: str, embedding_dim: int = 512):
        self.storage_path = storage_path
        self.dataset_path = os.path.join(self.storage_path, "aura_storage", "emotion_dataset")
        self.embedding_dim = embedding_dim

    def load_pair(self, hash_id: str) -> Tuple[np.ndarray, str, np.ndarray, str]:
        """
        Loads the four components (face image, caption, embedding, video path) 
        for a given hash ID from the dataset path.

        Args:
        - hash_id (str): The unique hash identifier for the pair.

        Returns:
        - Tuple containing the face image, caption, embedding, and video path.
        """
        folder_path = os.path.join(self.dataset_path, hash_id)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory for hash {hash_id} not found.")

        face_image_path = os.path.join(folder_path, "face.jpg")
        caption_path = os.path.join(folder_path, "caption.txt")
        embedding_path = os.path.join(folder_path, "embedding.npy")
        video_path = os.path.join(folder_path, "video.mp4")

        face_image = PairsProvider.load_image(face_image_path)
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        embedding = np.load(embedding_path)

        return face_image, caption, embedding, video_path
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Loads an image from the given path and returns it as a NumPy array.

        Args:
        - image_path (str): Path to the image file.

        Returns:
        - np.ndarray: The loaded image.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found at path: {image_path}")

        image = cv2.imread(image_path)
        return image

    @staticmethod
    def get_video_frames(video_path: str, as_numpy: bool = True) -> List[np.ndarray]:
        """
        Extracts frames from a video file.

        Args:
        - video_path (str): Path to the video file.
        - as_numpy (bool): If True, return frames as NumPy arrays. If False, convert to PyTorch tensors (optional).

        Returns:
        - List of frames as NumPy arrays.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if as_numpy:
                frames.append(frame)

        cap.release()
        return frames
    
    def load_embedding(self, embedding_path: str, verify_size: bool = False) -> np.ndarray:
        """
        Returns the numpy embedding vector from a .npy file.

        Args:
        - embedding_path (str): Path to the .npy embedding file.
        - verify_size (bool): Whether to verify that the embedding vector has the expected size.

        Returns:
        - np.ndarray: The loaded embedding vector.
        """
        if not os.path.isfile(embedding_path):
            raise FileNotFoundError(f"Embedding file not found at path: {embedding_path}")
        if not embedding_path.endswith('.npy'):
            raise ValueError(f"Invalid file type: {embedding_path}. Expected a .npy file.")

        embedding = np.load(embedding_path)

        if verify_size:
            assert embedding.shape[0] == self.embedding_dim, (
                f"Expected embedding of size {self.embedding_dim}, "
                f"but got size {embedding.shape[0]} at {embedding_path}. "
                f"Please verify the embedding size."
            )

        return embedding
    
    @staticmethod
    def get_first_video_frame(video_path: str, as_numpy: bool = True) -> np.ndarray:
        """
        Returns the first frame of the video.

        Args:
        - video_path (str): Path to the video file.
        - as_numpy (bool): If True, return the frame as a NumPy array.

        Returns:
        - The first frame as a NumPy array.
        """
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Failed to read the first frame from the video.")

        if as_numpy:
            return frame

    def get_all_hash_ids(self) -> List[str]:
        """
        Retrieves all hash IDs available in the dataset.

        Returns:
        - List of hash IDs.
        """
        return [name for name in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, name))]
