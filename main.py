#!/usr/bin/env python
"""
AI Art Generator - Main entry point
"""
import argparse
import logging
import os
import sys
import torch

from app.interface.gradio_ui import launch_ui
from config.settings import MODELS_DIR, DEVICE


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Art Generator")
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the Gradio interface on"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a publicly shareable link"
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Force CPU usage even if CUDA is available"
    )
    return parser.parse_args()


def main():
    """Main function to run the application."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    args = parse_args()
    
    # Ensure model directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Display environment info
    logger.info(f"PyTorch version: {torch.__version__}")
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("Forcing CPU mode as requested")
    elif torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"Using device: {DEVICE}")
    else:
        logger.warning("CUDA not available, using CPU. This will be significantly slower!")
        logger.warning("For image generation, consider using a smaller resolution and fewer steps.")
    
    # Launch the UI
    logger.info("Starting AI Art Generator...")
    try:
        launch_ui(port=args.port, share=args.share)
    except Exception as e:
        logger.error(f"Error starting the application: {e}")
        logger.info("Check your PyTorch installation and make sure dependencies are properly installed.")
        sys.exit(1)


if __name__ == "__main__":
    main()