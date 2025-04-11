"""
Tests for the style transfer module.
"""
import os
import pytest
from PIL import Image, ImageDraw
import numpy as np

from app.generator.style_transfer import StyleTransfer


@pytest.fixture
def style_transfer():
    """Fixture for style transfer instance."""
    return StyleTransfer()


@pytest.fixture
def sample_content_image():
    """Create a sample content image for testing."""
    # Create a simple geometric image
    img = Image.new('RGB', (128, 128), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([(20, 20), (108, 108)], fill='lightgray')
    draw.ellipse([(30, 30), (98, 98)], fill='gray')
    return img


@pytest.fixture
def sample_style_image():
    """Create a sample style image for testing."""
    # Create a simple colorful pattern
    img = Image.new('RGB', (128, 128), color='blue')
    draw = ImageDraw.Draw(img)
    for i in range(0, 128, 16):
        color = (i * 2, 255 - i * 2, i)
        draw.line([(0, i), (128, i)], fill=color, width=8)
    return img


def test_style_transfer_initialization(style_transfer):
    """Test that the style transfer module initializes correctly."""
    assert style_transfer.device is not None
    assert style_transfer.cnn is not None
    assert len(style_transfer.content_layers) > 0
    assert len(style_transfer.style_layers) > 0


def test_style_transfer_basic(style_transfer, sample_content_image, sample_style_image):
    """Test basic style transfer functionality.
    
    Note: This test is computationally intensive.
    It uses small images and few steps to speed up testing.
    """
    # Skip if CI environment is detected
    if os.environ.get("CI") == "true":
        pytest.skip("Skipping test in CI environment")
    
    # Apply style transfer with reduced steps for faster testing
    result = style_transfer.transfer_style(
        content_img=sample_content_image,
        style_img=sample_style_image,
        num_steps=10,  # Use very few steps for testing
        size=64,  # Use small size for faster testing
    )
    
    # Check that we got a valid image
    assert isinstance(result, Image.Image)
    assert result.size == (64, 64)
    assert result.mode == "RGB"
    
    # Check that the result is different from both inputs
    # Convert to numpy arrays for comparison
    content_array = np.array(sample_content_image.resize((64, 64)))
    style_array = np.array(sample_style_image.resize((64, 64)))
    result_array = np.array(result)
    
    content_diff = np.mean(np.abs(content_array - result_array))
    style_diff = np.mean(np.abs(style_array - result_array))
    
    # The result should differ from both inputs
    assert content_diff > 10  # Arbitrary threshold
    assert style_diff > 10  # Arbitrary threshold