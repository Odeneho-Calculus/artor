"""
Text-to-Image generation module using Stable Diffusion.
"""
import logging
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from config.settings import (
    DEVICE,
    PRECISION,
    SD_GUIDANCE_SCALE,
    SD_IMAGE_SIZE,
    SD_INFERENCE_STEPS,
    STABLE_DIFFUSION_MODEL,
)

logger = logging.getLogger(__name__)


class TextToImageGenerator:
    """Generator class for text-to-image conversion using Stable Diffusion."""

    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize the text-to-image generator.
        
        Args:
            model_id: The Hugging Face model ID or local path to the model.
                     If None, uses the default model from settings.
        """
        self.model_id = model_id or STABLE_DIFFUSION_MODEL
        self.device = DEVICE
        logger.info(f"Text-to-Image generator using device: {self.device}")
        self.pipeline = None

    def load_model(self):
        """Load the Stable Diffusion model."""
        logger.info(f"Loading Stable Diffusion model '{self.model_id}' on {self.device}")
        
        # Determine torch dtype based on device and precision setting
        torch_dtype = torch.float16 if self.device == "cuda" and PRECISION == "fp16" else torch.float32
        
        try:
            # For CPU, we'll need to be more careful with memory usage
            safety_kwargs = {}
            memory_kwargs = {}
            
            if self.device == "cpu":
                logger.info("Using CPU optimizations for Stable Diffusion")
                # No need for safety checker on CPU to save memory
                safety_kwargs["safety_checker"] = None
            else:
                # For GPU, disable safety checker for speed
                safety_kwargs["safety_checker"] = None
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                **safety_kwargs
            )
            
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory optimizations if available
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
            
            # Additional CPU optimizations
            if self.device == "cpu":
                if hasattr(self.pipeline, "enable_sequential_cpu_offload"):
                    self.pipeline.enable_sequential_cpu_offload()
                elif hasattr(self.pipeline, "enable_model_cpu_offload"):
                    self.pipeline.enable_model_cpu_offload()
                    
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = SD_IMAGE_SIZE,
        height: int = SD_IMAGE_SIZE,
        num_inference_steps: int = SD_INFERENCE_STEPS,
        guidance_scale: float = SD_GUIDANCE_SCALE,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: The text prompt to generate an image from.
            negative_prompt: Text to discourage in the generation.
            width: Width of the generated image.
            height: Height of the generated image.
            num_inference_steps: Number of denoising steps.
            guidance_scale: How closely to follow the prompt.
            seed: Random seed for reproducibility.
            
        Returns:
            Generated PIL Image.
        """
        if self.pipeline is None:
            self.load_model()
            
        logger.info(f"Generating image for prompt: '{prompt}'")
        
        # Set the random seed if provided
        generator = None
        if seed is not None:
            # Make sure we use the correct device for the generator
            generator = torch.Generator(device=self.device if self.device == "cuda" else "cpu").manual_seed(seed)
            logger.info(f"Using seed: {seed}")
        
        # CPU-specific adjustments
        if self.device == "cpu":
            # Use fewer steps on CPU for reasonable speed
            if num_inference_steps > 30:
                logger.info(f"Reducing inference steps from {num_inference_steps} to 30 for CPU")
                num_inference_steps = 30
            
            # Cap image size for CPU
            if width > 512 or height > 512:
                logger.info(f"Reducing image size from {width}x{height} to 512x512 for CPU")
                width = min(width, 512)
                height = min(height, 512)
        
        # Generate the image
        try:
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            
            # Return the first generated image
            image = result.images[0]
            logger.info("Image generated successfully")
            return image
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise