"""
Tests for the generator module.
"""
import os
import pytest
from PIL import Image

from app.generator.text_to_image import TextToImageGenerator
from config.settings import STABLE_DIFFUSION_MODEL


@pytest.fixture
def text_to_image_generator():
    """Fixture for text-to-image generator instance."""
    generator = TextToImageGenerator()
    return generator


def test_generator_initialization(text_to_image_generator):
    """Test that the generator initializes correctly."""
    assert text_to_image_generator.model_id == STABLE_DIFFUSION_MODEL
    assert text_to_image_generator.pipeline is None


def test_generate_simple_image(text_to_image_generator):
    """Test basic image generation.
    
    Note: This test requires downloading the model and is slow.
    It might be skipped in CI environments.
    """
    # Skip if CI environment is detected
    if os.environ.get("CI") == "true":
        pytest.skip("Skipping test in CI environment")
    
    # Generate a simple image
    prompt = "A photo of a cat"
    image = text_to_image_generator.generate(
        prompt=prompt,
        width=256,  # Use small size for faster testing
        height=256,
        num_inference_steps=10,  # Use fewer steps for faster testing
    )
    
    # Check that we got a valid image
    assert isinstance(image, Image.Image)
    assert image.size == (256, 256)
    assert image.mode == "RGB"


def test_generate_with_seed(text_to_image_generator):
    """Test that seed parameter produces deterministic results.
    
    Note: This test requires downloading the model and is slow.
    It might be skipped in CI environments.
    """
    # Skip if CI environment is detected
    if os.environ.get("CI") == "true":
        pytest.skip("Skipping test in CI environment")
    
    # Generate two images with the same seed
    prompt = "A photo of a mountain"
    seed = 42
    
    image1 = text_to_image_generator.generate(
        prompt=prompt,
        width=256,
        height=256,
        num_inference_steps=10,
        seed=seed,
    )
    
    image2 = text_to_image_generator.generate(
        prompt=prompt,
        width=256,
        height=256,
        num_inference_steps=10,
        seed=seed,
    )
    
    # Convert to numpy arrays for comparison
    import numpy as np
    array1 = np.array(image1)
    array2 = np.array(image2)
    
    # Verify images are identical (or extremely similar)
    # Allow small differences due to non-deterministic GPU operations
    assert np.mean(np.abs(array1 - array2)) < 1.0