"""
Utility functions for loading and managing models.
"""
import logging
import os
from typing import Optional, Dict, Any

import torch
from config.settings import MODELS_DIR, DEVICE

logger = logging.getLogger(__name__)


class ModelManager:
    """Manager class for handling model loading and caching."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.loaded_models = {}
        self.device = torch.device(DEVICE)
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def get_model(
        self,
        model_name: str,
        model_type: str,
        **model_args
    ) -> Any:
        """
        Load a model, either from cache or from disk/hub.
        
        Args:
            model_name: Name or path of the model
            model_type: Type of model to load (e.g., "stable-diffusion", "style-transfer")
            **model_args: Arguments to pass to the model loader
            
        Returns:
            Loaded model
        """
        model_key = f"{model_type}:{model_name}"
        
        # Check if model is already loaded
        if model_key in self.loaded_models:
            logger.info(f"Using cached model: {model_key}")
            return self.loaded_models[model_key]
        
        # Load the model based on type
        logger.info(f"Loading model: {model_key}")
        
        try:
            model = self._load_model_by_type(model_name, model_type, **model_args)
            
            # Cache the model
            self.loaded_models[model_key] = model
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            raise
    
    def _load_model_by_type(
        self,
        model_name: str,
        model_type: str,
        **model_args
    ) -> Any:
        """
        Load a model based on its type.
        
        Args:
            model_name: Name or path of the model
            model_type: Type of model to load
            **model_args: Arguments to pass to the model loader
            
        Returns:
            Loaded model
        """
        if model_type == "stable-diffusion":
            from diffusers import StableDiffusionPipeline
            
            # Determine torch dtype based on device
            dtype = torch.float16 if str(self.device) == "cuda" else torch.float32
            
            # Load the pipeline
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                safety_checker=None,  # Disable for speed
                **model_args
            )
            
            # Move to device and optimize
            pipeline = pipeline.to(self.device)
            
            # Apply optimizations
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            
            # Apply CPU optimizations if needed
            if str(self.device) == "cpu":
                if hasattr(pipeline, "enable_sequential_cpu_offload"):
                    pipeline.enable_sequential_cpu_offload()
                elif hasattr(pipeline, "enable_model_cpu_offload"):
                    pipeline.enable_model_cpu_offload()
            
            return pipeline
            
        elif model_type == "vgg":
            from torchvision import models
            
            # Load VGG model
            vgg = models.vgg19(weights="DEFAULT").features.to(self.device).eval()
            return vgg
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def clear_cache(self):
        """Clear the model cache to free memory."""
        self.loaded_models.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Model cache cleared")


def get_model_cache() -> ModelCache:
    """Get the model cache singleton instance."""
    return ModelCache()


def download_model(model_id: str, variant: str = None) -> str:
    """
    Download a model to the local models directory.
    
    Args:
        model_id: Hugging Face model ID
        variant: Model variant (if applicable)
        
    Returns:
        Path to the downloaded model
    """
    from huggingface_hub import snapshot_download
    
    # Create target directory
    local_dir = os.path.join(MODELS_DIR, model_id.replace("/", "_"))
    os.makedirs(local_dir, exist_ok=True)
    
    logger.info(f"Downloading model {model_id} to {local_dir}")
    
    # Download the model
    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            revision=variant
        )
        logger.info(f"Model downloaded to {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def get_available_models() -> Dict[str, str]:
    """
    Get a dictionary of locally available models.
    
    Returns:
        Dictionary mapping model names to paths
    """
    models = {}
    
    # Check if models directory exists
    if not os.path.exists(MODELS_DIR):
        return models
        
    # Check for subdirectories in the models directory
    for item in os.listdir(MODELS_DIR):
        item_path = os.path.join(MODELS_DIR, item)
        if os.path.isdir(item_path):
            # Convert underscore-separated path back to model ID format
            model_id = item.replace("_", "/")
            models[model_id] = item_path
            
    return models