import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from aura.dataset import DatasetProvider

@pytest.fixture
def dataset_provider():
    with patch('kagglehub.dataset_download') as mock_download:
        mock_download.return_value = "mock/path"
        with patch.object(DatasetProvider, '_collect_files') as mock_collect:
            mock_data = [(np.zeros((64, 64, 3)), 0, "happy") for _ in range(10)]
            mock_collect.return_value = mock_data
            provider = DatasetProvider(target_size=(224, 224), split = 0.7)
            return provider
        
def test_collect_files_with_augmentation(dataset_provider):
    mock_files = [("mock/path", [], ["image1.jpg", "image2.jpg"])]
    with patch("os.walk", return_value=mock_files):
        with patch.object(dataset_provider, '_get_emotion', return_value="happy"):
            with patch('cv2.imread', return_value=np.zeros((64, 64, 3))):
                dataset = dataset_provider._collect_files("mock/path", augment=True)
                assert len(dataset) == 8  
                assert all(len(item) == 3 for item in dataset)
                first_img, first_label, first_emotion = dataset[0]
                assert isinstance(first_img, np.ndarray)
                assert isinstance(first_label, int)
                assert isinstance(first_emotion, str)

def test_collect_files_without_augmentation(dataset_provider):
    mock_files = [("mock/path", [], ["image1.jpg", "image2.jpg"])]
    with patch("os.walk", return_value=mock_files):
        with patch.object(dataset_provider, '_get_emotion', return_value="happy"):
            with patch('cv2.imread', return_value=np.zeros((64, 64, 3))):
                dataset = dataset_provider._collect_files("mock/path", augment=False)
                assert len(dataset) == 2  
                assert all(len(item) == 3 for item in dataset)
                first_img, first_label, first_emotion = dataset[0]
                assert isinstance(first_img, np.ndarray)
                assert isinstance(first_label, int)
                assert isinstance(first_emotion, str)

def test_init(dataset_provider):
    assert dataset_provider.target_size == (224, 224)
    assert len(dataset_provider.emotion_labels) == 8
    assert len(dataset_provider.train) == 8 
    assert len(dataset_provider.test) == 2   

def test_sample_valid_index(dataset_provider):
    img, label, emotion = dataset_provider.sample(0, source = "test")
    assert isinstance(img, np.ndarray)
    assert img.shape == (64, 64, 3)
    assert isinstance(label, int)
    assert 0 <= label <= 7
    assert emotion in dataset_provider.emotion_labels

def test_sample_invalid_index(dataset_provider):
    with pytest.raises(ValueError):
        dataset_provider.sample(100, source = "train")

def test_get_next_image_batch(dataset_provider):
    batch_size = 2
    batch_generator = dataset_provider.get_next_image_batch(batch_size, source = "train")
    first_batch = next(batch_generator)
    
    assert len(first_batch) == batch_size
    assert isinstance(first_batch[0][0], np.ndarray)
    assert isinstance(first_batch[0][1], int)

def test_get_next_image_batch_invalid_size(dataset_provider):
    with pytest.raises(ValueError):
        next(dataset_provider.get_next_image_batch(100, source = "train"))

def test_resize_image(dataset_provider):
    test_image = np.zeros((100, 150, 3), dtype=np.uint8)
    resized = dataset_provider._resize_image(test_image)
    assert resized.shape == (224, 224, 3)

def test_random_rotation(dataset_provider):
    test_image = np.zeros((64, 64, 3), dtype=np.uint8)
    rotated = dataset_provider._random_rotation(test_image)
    assert rotated.shape == test_image.shape

def test_flip(dataset_provider):
    test_image = np.zeros((64, 64, 3), dtype=np.uint8)
    flipped = dataset_provider._flip(test_image)
    assert flipped.shape == test_image.shape

def test_random_brightness(dataset_provider):
    test_image = np.zeros((64, 64, 3), dtype=np.uint8)
    brightened = dataset_provider._random_brightness(test_image)
    assert brightened.shape == test_image.shape

def test_get_emotion_invalid_path(dataset_provider):
    with pytest.raises(ValueError):
        dataset_provider._get_emotion("invalid_path")

def test_set_black_background(dataset_provider):
    test_image = np.ones((64, 64, 3), dtype=np.uint8) * 255
    processed = dataset_provider._set_black_background(test_image, threshold=20)
    assert processed.shape == test_image.shape
    assert processed.dtype == np.uint8
