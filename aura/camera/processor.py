from typing import Optional, Tuple, Union
import os
import cv2
import numpy as np
import logging
from datetime import datetime
import time

class FaceNotFoundException(Exception):
    """Exception raised when no face is detected in an image."""
    def __init__(self, message: str = "Face not found in the image") -> None:
        super().__init__(message)

class ProcessingPipeline:
    """Pipeline for processing images to detect and extract faces."""
    
    def __init__(self, log_path: str = "./logs", verbose: int = 0) -> None:
        """
        Initialize the processing pipeline.
        
        Args:
            log_path: Directory path for storing logs and processed images
            verbose: Logging level (0: no logging, 1: save images, 2: save images and timing logs)
        
        Raises:
            ValueError: If verbose is not in range [0, 2]
        """
        if verbose > 2:
            raise ValueError("ProcessingPipeline verbosity can only be between 0 and 2.")
        
        self.log_path: str = log_path
        self.verbose: int = verbose
        self.face_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.current_log_dir: Optional[str] = None
        
        if self.verbose > 0:
            self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging directory and settings based on verbosity level."""
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_dir = os.path.join(self.log_path, timestamp)
        os.makedirs(self.current_log_dir, exist_ok=True)
        
        if self.verbose == 2:
            logging.basicConfig(
                filename=os.path.join(self.current_log_dir, 'processing.log'),
                level=logging.INFO,
                format='%(asctime)s - %(message)s'
            )
    
    def _save_to_log_path(self, image: np.ndarray, filename: str) -> None:
        """
        Save an image to the logging directory.
        
        Args:
            image: Image array to save
            filename: Name of the output file
        """
        if self.verbose > 0:
            if not hasattr(self, 'current_log_dir'):
                self._setup_logging()
            cv2.imwrite(os.path.join(self.current_log_dir, filename), image)
    
    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        """
        Convert an image to grayscale if it's not already.
        
        Args:
            image: Input image array (BGR or grayscale)
            
        Returns:
            Grayscale version of the input image
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    def get_bounding_box(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the largest face in an image and return its bounding box.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (x, y, width, height) for the largest face, or None if no face detected
        """
        start_time: float = time.time()
        gray: np.ndarray = self._convert_to_gray(image)
        faces: np.ndarray = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if self.verbose == 2:
            logging.info(f"Face detection took {time.time() - start_time:.2f} seconds")
        
        if len(faces) == 0:
            return None
            
        return sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]
    
    def annotate_face(self, image: np.ndarray) -> np.ndarray:
        """
        Draw a bounding box around the largest detected face.
        
        Args:
            image: Input image array
            
        Returns:
            Image with annotated face bounding box
        """
        bbox: Optional[Tuple[int, int, int, int]] = self.get_bounding_box(image)
        if bbox is not None:
            x, y, w, h = bbox
            annotated_image: np.ndarray = image.copy()
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            self._save_to_log_path(annotated_image, "bbox.jpg")
            return annotated_image
        return image
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Extract, resize, and normalize the largest face in an image.
        
        Args:
            image: Input image array
            
        Returns:
            48x48 normalized grayscale face image
            
        Raises:
            FaceNotFoundException: If no face is detected in the image
        """
        start_time: float = time.time()
        gray: np.ndarray = self._convert_to_gray(image)
        bbox: Optional[Tuple[int, int, int, int]] = self.get_bounding_box(gray)
        
        if bbox is None:
            raise FaceNotFoundException()
        
        x, y, w, h = bbox
        face: np.ndarray = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype(np.float32) / 255.0
        
        if self.verbose > 0:
            self._save_to_log_path((face * 255).astype(np.uint8), "processed.jpg")
        
        if self.verbose == 2:
            logging.info(f"Image processing took {time.time() - start_time:.2f} seconds")
        
        return face
