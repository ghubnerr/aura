import os
import random
from typing import Union, List

import cv2
import kagglehub
import numpy as np

class DatasetProvider:
    def __init__(self, save_dir : Union[None, str]):
        datasets = []
        datasets.append(kagglehub.dataset_download("jafarhussain786/human-emotionshappy-faces"), path = save_dir)
        datasets.append(kagglehub.dataset_download("jafarhussain786/human-emotionssad-faces"), path = save_dir)
        datasets.append(kagglehub.dataset_download("jafarhussain786/human-emotionsangry-faces"), path = save_dir)
        datasets.append(kagglehub.dataset_download("jafarhussain786/human-emotionsfear-faces"), path = save_dir)
        datasets.append(kagglehub.dataset_download("jafarhussain786/human-emotionssuprise-faces"), path = save_dir)

        self.image_paths = []
        for path in datasets:
            self.image_paths.extend(DatasetProvider._collect_files(path))

    def get_image(self, **kwargs) -> cv2.MatLike:
        """
        Returns a random, unprocessed image from our dataset
        """
        path = random.choice(self.image_paths)
        image = None
        try:
            image = cv2.imread(path)
        except:
            print(f"Could not read file from path: {path}")
            return None

        return DatasetProvider._resize_image(image)

    def get_next_image_batch(self, **kwargs) -> List[cv2.MatLike]:
        """
        Returns groups of images (positives) to train SimCLR embeddings.
        """
        image = DatasetProvider.get_image()
        image = DatasetProvider._set_black_background(image)

        batch = []
        batch.append(DatasetProvider._flip(image))
        batch.append(DatasetProvider._random_brightness(image))
        batch.append(DatasetProvider._random_zoom(image))
        batch.append(DatasetProvider._random_rotation(image))

        return batch

    def _collect_files(path: str) -> List[str]:
        image_path = []
        for f in os.listdir(path):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                image_path.append(os.path.join(path, f))
        return image_path 

    
    def _resize_image(image: cv2.MatLike, target_size=(224, 224)):
        # If the image is grayscale, convert it to 3 channels (RGB)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = image.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized_image = cv2.resize(image, (new_w, new_h))
        canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        top = (target_size[0] - new_h) // 2
        left = (target_size[1] - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized_image
        return canvas
    
    def _random_rotation(image: cv2.MatLike):
        angle = np.random.uniform(-30, 30)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    def _flip(image: cv2.MatLike):
        return cv2.flip(image, 1)  
    
    def _random_brightness(image: cv2.MatLike):
        factor = np.random.uniform(0.5, 1.5)
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def _random_zoom(image: cv2.MatLike):
        h, w = image.shape[:2]
        zoom = np.random.uniform(0.3, 2.0)  
        new_h, new_w = int(h * zoom), int(w * zoom)
        image = cv2.resize(image, (new_w, new_h))
        if new_h < h or new_w < w:
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            cropped = image[top:top + new_h, left:left + new_w]
        else:
            cropped = image[:h, :w]
        return DatasetProvider.resize_image(cropped)
    
    def _set_black_background(image, threshold=20):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)
        image[mask == 255] = [0, 0, 0]
        return image