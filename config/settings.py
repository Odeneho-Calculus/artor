"""
Global settings for the Artor.
"""
import os
import torch
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory to store models
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()

# Device configuration - automatically use CPU if CUDA is not available
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
PRECISION = "fp16" if DEVICE == "cuda" else "fp32"

# Stable Diffusion settings
STABLE_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"  # Default model
SD_IMAGE_SIZE = 512  # Default size for generated images
SD_INFERENCE_STEPS = 50
SD_GUIDANCE_SCALE = 7.5

# Style transfer settings
STYLE_WEIGHT = 1000000.0
CONTENT_WEIGHT = 1.0
NUM_STYLE_TRANSFER_STEPS = 300

# UI settings
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MAX_UPLOAD_SIZE = 5  # in MB