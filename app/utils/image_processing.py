"""
Utility functions for image processing.
"""
import io
import logging
import os
from typing import Tuple, Optional, List

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


def resize_image(image: Image.Image, size: Tuple[int, int], keep_aspect: bool = True) -> Image.Image:
    """
    Resize an image while optionally maintaining aspect ratio.
    
    Args:
        image: PIL Image to resize
        size: (width, height) tuple
        keep_aspect: If True, preserve aspect ratio and fill with black
        
    Returns:
        Resized PIL Image
    """
    if not keep_aspect:
        return image.resize(size, Image.LANCZOS)
    
    # Resize keeping aspect ratio
    img_ratio = image.width / image.height
    target_ratio = size[0] / size[1]
    
    if img_ratio > target_ratio:
        # Image is wider than target
        new_height = int(size[0] / img_ratio)
        resized = image.resize((size[0], new_height), Image.LANCZOS)
        
        # Create black canvas
        new_img = Image.new("RGB", size, (0, 0, 0))
        paste_y = (size[1] - new_height) // 2
        new_img.paste(resized, (0, paste_y))
        
    else:
        # Image is taller than target
        new_width = int(size[1] * img_ratio)
        resized = image.resize((new_width, size[1]), Image.LANCZOS)
        
        # Create black canvas
        new_img = Image.new("RGB", size, (0, 0, 0))
        paste_x = (size[0] - new_width) // 2
        new_img.paste(resized, (paste_x, 0))
    
    return new_img


def apply_basic_enhancements(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0,
) -> Image.Image:
    """
    Apply basic image enhancements.
    
    Args:
        image: PIL Image to enhance
        brightness: Brightness factor (1.0 = original)
        contrast: Contrast factor (1.0 = original)
        saturation: Saturation factor (1.0 = original)
        sharpness: Sharpness factor (1.0 = original)
        
    Returns:
        Enhanced PIL Image
    """
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    
    if saturation != 1.0:
        image = ImageEnhance.Color(image).enhance(saturation)
    
    if sharpness != 1.0:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
    
    return image


def apply_filter(image: Image.Image, filter_type: str) -> Image.Image:
    """
    Apply a filter to an image.
    
    Args:
        image: PIL Image to filter
        filter_type: Type of filter to apply (blur, contour, sharpen, etc.)
        
    Returns:
        Filtered PIL Image
    """
    filter_map = {
        "blur": ImageFilter.BLUR,
        "contour": ImageFilter.CONTOUR,
        "edge_enhance": ImageFilter.EDGE_ENHANCE,
        "edge_enhance_more": ImageFilter.EDGE_ENHANCE_MORE,
        "emboss": ImageFilter.EMBOSS,
        "find_edges": ImageFilter.FIND_EDGES,
        "sharpen": ImageFilter.SHARPEN,
        "smooth": ImageFilter.SMOOTH,
        "smooth_more": ImageFilter.SMOOTH_MORE,
    }
    
    if filter_type not in filter_map:
        logger.warning(f"Unknown filter type: {filter_type}")
        return image
    
    return image.filter(filter_map[filter_type])


def save_image_with_metadata(
    image: Image.Image,
    filepath: str,
    prompt: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> str:
    """
    Save an image with metadata embedded.
    
    Args:
        image: PIL Image to save
        filepath: Path to save the image to
        prompt: Text prompt used to generate the image
        parameters: Dictionary of generation parameters
        
    Returns:
        Path to the saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Add metadata to the image
    metadata = {}
    
    if prompt:
        metadata["prompt"] = prompt
    
    if parameters:
        for key, value in parameters.items():
            metadata[f"param:{key}"] = str(value)
    
    # Add timestamp
    import datetime
    metadata["generated_at"] = datetime.datetime.now().isoformat()
    
    # Save with metadata
    image.save(filepath, format="PNG", pnginfo=_create_pnginfo(metadata))
    logger.info(f"Image saved to {filepath}")
    
    return filepath


def _create_pnginfo(metadata_dict: dict):
    """Create PNG metadata from dictionary."""
    from PIL import PngImagePlugin
    
    metadata = PngImagePlugin.PngInfo()
    
    for key, value in metadata_dict.items():
        metadata.add_text(key, str(value))
    
    return metadata


def pil_to_base64(image: Image.Image) -> str:
    """
    Convert a PIL Image to a base64 string.
    
    Args:
        image: PIL Image to convert
        
    Returns:
        Base64 encoded string of the image
    """
    import base64
    
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def base64_to_pil(base64_str: str) -> Image.Image:
    """
    Convert a base64 string to a PIL Image.
    
    Args:
        base64_str: Base64 encoded string of an image
        
    Returns:
        PIL Image
    """
    import base64
    
    # Remove header if present
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    
    img_data = base64.b64decode(base64_str)
    buffer = io.BytesIO(img_data)
    return Image.open(buffer)


def create_image_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        images: List of PIL Images
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        
    Returns:
        PIL Image of the grid
    """
    if len(images) > rows * cols:
        logger.warning(f"Too many images ({len(images)}) for grid size ({rows}x{cols})")
        images = images[:rows * cols]
    
    # Make sure all images have the same size
    width, height = images[0].size
    
    # Create the grid
    grid = Image.new('RGB', (cols * width, rows * height))
    
    for i, img in enumerate(images):
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        
        row = i // cols
        col = i % cols
        grid.paste(img, (col * width, row * height))
    
    return grid