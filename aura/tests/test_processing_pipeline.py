import pytest
import numpy as np
import cv2
import os
import logging
from pathlib import Path
import tempfile
import requests
from io import BytesIO
from aura.camera import ProcessingPipeline, FaceNotFoundException


def download_image(url):
    response = requests.get(url)
    return cv2.imdecode(
        np.frombuffer(response.content, np.uint8),
        cv2.IMREAD_COLOR
    )

@pytest.fixture
def pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield ProcessingPipeline(log_path=tmp_dir, verbose=2)

@pytest.fixture
def image_with_face():
    url = "https://www.yourtango.com/sites/default/files/image_blog/habits-of-truly-nice-people.png"
    return download_image(url)

@pytest.fixture
def image_without_face():
    url = "https://www.bsr.org/images/heroes/bsr-focus-nature-hero.jpg"
    return download_image(url)

def test_initialization(pipeline):
    assert pipeline.verbose == 2
    assert os.path.exists(pipeline.log_path)
    assert pipeline.face_cascade is not None

def test_invalid_verbosity():
    with pytest.raises(ValueError):
        ProcessingPipeline(verbose=3)

def test_face_detection_with_face(pipeline, image_with_face):
    bbox = pipeline.get_bounding_box(image_with_face)
    assert bbox is not None
    assert len(bbox) == 4
    x, y, w, h = bbox
    assert all(isinstance(val, (int, np.int32, np.int64)) for val in [x, y, w, h])
    assert w > 0 and h > 0

def test_face_detection_without_face(pipeline, image_without_face):
    bbox = pipeline.get_bounding_box(image_without_face)
    assert bbox is None

def test_annotation(pipeline, image_with_face):
    annotated = pipeline.annotate_face(image_with_face)
    assert annotated.shape == image_with_face.shape
    # Check if the log file was created
    log_files = list(Path(pipeline.current_log_dir).glob("bbox.jpg"))
    assert len(log_files) == 1

def test_image_processing_with_face(pipeline, image_with_face):
    processed = pipeline.process_image(image_with_face)
    assert processed.shape == (48, 48)
    assert processed.dtype == np.float32
    assert 0 <= processed.min() <= processed.max() <= 1
    log_files = list(Path(pipeline.current_log_dir).glob("processed.jpg"))
    assert len(log_files) == 1

def test_image_processing_without_face(pipeline, image_without_face):
    with pytest.raises(FaceNotFoundException):
        pipeline.process_image(image_without_face)

def test_grayscale_conversion(pipeline, image_with_face):
    gray = pipeline._convert_to_gray(image_with_face)
    assert len(gray.shape) == 2
    gray2 = pipeline._convert_to_gray(gray)
    assert np.array_equal(gray, gray2)
