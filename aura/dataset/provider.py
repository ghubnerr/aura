import os
import random
from typing import Union, List
from collections import defaultdict, Counter

import cv2
import kagglehub
import numpy as np

CLASS_CUTOFF = 2000

class DatasetProvider:
    def __init__(self, target_size: Union[None, tuple[int, int]] = None, augment: bool = False):
        self.target_size = target_size
        self.emotion_labels = {
            "happy": 0,
            "sad": 1,
            "anger": 2,
            "disgust": 3,
            "fear": 4,
            "neutral": 5,
            "surprise": 6,
            "contempt": 7,
        }

        original_dir = os.getcwd()
        try:
            save_dir = os.path.join(os.environ.get("STORAGE_PATH"), "aura_storage", "emotion_dataset")
            os.makedirs(save_dir, exist_ok = True)
            os.chdir(save_dir)
        except:
            print(f"Could not access to aura storage, {os.environ.get('STORAGE_PATH')}, saving to process directory.")

        dataset_path = kagglehub.dataset_download("noamsegal/affectnet-training-data")
        os.chdir(original_dir)

        dataset = self._collect_files(dataset_path, augment)

        train_size = int(len(dataset) * .8)

        self.train = dataset[:train_size]
        self.test = dataset[train_size:]

    def sample(self, index: int, test: bool) -> tuple[np.ndarray, int, str]:
        """
        Returns an image, its respective label, and emotion given an index
        """
        dataset = self.test if test else self.train
        if index >= len(dataset):
            raise ValueError(f"The index provided of value {index} is larger than the length of dataset, {len(dataset)}")
        
        return dataset[index]

    def get_next_image_batch(self, batch_size: int, test: bool):
        """
        Generator that yields batches of images from the dataset.
        """
        dataset = self.test if test else self.train
        if batch_size > len(dataset):
            raise ValueError("Can't get batch size larger than dataset")

        indices = list(range(len(dataset)))
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = []
            for idx in batch_indices:
                batch.append(self.sample(idx, test))
            yield batch

    def label_distr(self, test = False):
        dataset = self.test if test else self.train
        counter = Counter([emotion for _, _, emotion in dataset])
        for emotion, count in counter.items():
            print(f"{emotion}: {(count/len(dataset))*100:.2f}%")
    
    def _get_emotion(self, path):
        dirs = path.split("/")
        if len(dirs) < 2:
            raise ValueError(f"Path, {path}, is not in proper format to parse emotion")
        return dirs[-2]

    def _collect_files(self, path: str, augment: bool = False) -> List[str]:
        dataset = []
        label_counter = defaultdict(int)
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                img_path = os.path.join(dirpath, filename)
                if not img_path.endswith((".jpg", ".png", ".jpeg")):
                    continue

                emotion = self._get_emotion(img_path)
                if label_counter[emotion] >= 2000:
                    continue
                
                label = self.emotion_labels[emotion]
                img = cv2.imread(img_path)

                if augment:
                    rot_img = self._random_rotation(img)
                    flip_img = self._flip(img)
                    bright_img = self._random_brightness(img)
                    mods = [img, rot_img, flip_img, bright_img]
                else:
                    mods = [img]

                for mod in mods:
                    dataset.append((mod, label, emotion))

                label_counter[self._get_emotion(img_path)] += len(mods)

        random.shuffle(dataset)
        return dataset

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if self.target_size is None:
            print("Called `_resize_image` without set target_size.")
            return

        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array.")
        
        if len(image.shape) not in [2, 3]:
            raise ValueError("Input image must be either grayscale (2D) or RGB/BGR (3D).")
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            print("Image shape", image.shape)
            print("h, w", h, w)
            raise ValueError("Input image has invalid dimensions (height or width is zero).")
        
        scale = min(self.target_size[0] / h, self.target_size[1] / w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        canvas = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        top = (self.target_size[0] - new_h) // 2
        left = (self.target_size[1] - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized_image
        
        output_image = np.ascontiguousarray(canvas, dtype=np.uint8)
        return output_image
    
    def _random_rotation(self, image: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(-30, 30)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    def _flip(self, image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 1)  
    
    def _random_brightness(self, image: np.ndarray) -> np.ndarray:
        factor = np.random.uniform(0.5, 1.5)
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def _random_zoom(self, image: np.ndarray) -> np.ndarray:
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
        return self._resize_image(cropped)
    
    def _set_black_background(self, image: np.ndarray, threshold=20) -> np.ndarray:
        image = np.ascontiguousarray(image, dtype=np.uint8)  # Ensure image is contiguous and of type uint8
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)
        image[mask == 255] = [0, 0, 0]
        return image