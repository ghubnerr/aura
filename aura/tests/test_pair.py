import pytest
from PIL import Image
import subprocess
from aura.dataset import PairsGenerator
import time

@pytest.fixture
def random_image():
    """Generate a random image for testing."""
    width, height = 100, 100
    image = Image.new("RGB", (width, height), "blue") 
    return image


@pytest.fixture
def ensure_ollama_running():
    """Ensure Ollama server is running, skip test if it's not available."""
    try:
        subprocess.run(["ollama", "list"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        pytest.skip("Ollama CLI is not installed.")
    except subprocess.CalledProcessError:
        pytest.skip("Ollama is not running or not connected to the app.")
        
def test_generate_text_description(random_image, ensure_ollama_running):
    """
    Test _generate_text_description using Ollama. Skips if Ollama is unavailable.
    """
    description = PairsGenerator._generate_text_description(random_image, model="ollama/bakllava")

    assert isinstance(description, str)
    assert len(description) > 0
    print(f"Generated description: {description}")
