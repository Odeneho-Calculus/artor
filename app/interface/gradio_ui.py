"""
Gradio-based user interface for the AI Art Generator.
"""
import io
import logging
import os
import tempfile
from typing import Tuple

import gradio as gr
import PIL
from PIL import Image

from app.generator.style_transfer import StyleTransfer
from app.generator.text_to_image import TextToImageGenerator
from config.settings import DEFAULT_OUTPUT_DIR, MAX_UPLOAD_SIZE

logger = logging.getLogger(__name__)


class ArtGeneratorUI:
    """Class to handle the Gradio UI for the AI Art Generator."""
    
    def __init__(self):
        """Initialize the UI components and models."""
        self.text_to_image_generator = TextToImageGenerator()
        self.style_transfer = StyleTransfer()
        
        # Ensure output directory exists
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        
        logger.info("UI components initialized")
    
    def _generate_from_text(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Image.Image:
        """Generate an image from text prompt."""
        if not prompt:
            raise gr.Error("Please enter a text prompt")
            
        try:
            # Handle seed value (0 means random)
            seed = None if seed == 0 else seed
            
            # Generate the image
            return self.text_to_image_generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        except Exception as e:
            logger.error(f"Error in text-to-image generation: {e}")
            raise gr.Error(f"Failed to generate image: {e}")
    
    def _apply_style_transfer(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        strength: float,
        steps: int,
    ) -> Image.Image:
        """Apply style transfer to combine content and style."""
        if content_image is None:
            raise gr.Error("Please provide a content image")
        if style_image is None:
            raise gr.Error("Please provide a style image")
            
        try:
            # Adjust weights based on strength
            style_weight = strength * 1000000.0
            content_weight = 1.0
            
            # Apply style transfer
            return self.style_transfer.transfer_style(
                content_img=content_image,
                style_img=style_image,
                num_steps=steps,
                style_weight=style_weight,
                content_weight=content_weight,
            )
        except Exception as e:
            logger.error(f"Error in style transfer: {e}")
            raise gr.Error(f"Style transfer failed: {e}")
    
    def _combined_generation(
        self,
        prompt: str,
        negative_prompt: str,
        style_image: Image.Image,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        style_strength: float,
        style_steps: int,
    ) -> Image.Image:
        """Generate an image from text and apply style transfer."""
        if not prompt:
            raise gr.Error("Please enter a text prompt")
        if style_image is None:
            raise gr.Error("Please provide a style image")
            
        try:
            # First generate the base image from text
            logger.info("Step 1: Generating base image from text")
            content_image = self._generate_from_text(
                prompt, negative_prompt, width, height, steps, guidance_scale, seed
            )
            
            # Then apply style transfer
            logger.info("Step 2: Applying style transfer")
            return self._apply_style_transfer(
                content_image, style_image, style_strength, style_steps
            )
        except Exception as e:
            logger.error(f"Error in combined generation: {e}")
            raise gr.Error(f"Generation failed: {e}")
    
    def _save_image(self, image: Image.Image) -> str:
        """Save an image to the output directory and return the path."""
        if image is None:
            return None
            
        # Create a unique filename
        import time
        timestamp = int(time.time())
        filename = f"ai_art_{timestamp}.png"
        filepath = os.path.join(DEFAULT_OUTPUT_DIR, filename)
        
        # Save the image
        image.save(filepath)
        logger.info(f"Image saved to {filepath}")
        
        return filepath

    def build_interface(self):
        """Build and return the Gradio interface."""
        with gr.Blocks(title="AI Art Generator") as interface:
            gr.Markdown("""
            # 🎨 AI Art Generator
            
            Create stunning AI-generated artwork using text prompts and style transfer.
            """)
            
            with gr.Tabs():
                # Tab 1: Text-to-Image
                with gr.TabItem("Text to Image"):
                    with gr.Row():
                        with gr.Column():
                            text_prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="Describe the image you want to generate...",
                                lines=3,
                            )
                            negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="What to avoid in the image...",
                                lines=2,
                            )
                            
                            with gr.Row():
                                width = gr.Slider(
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=64,
                                    label="Width",
                                )
                                height = gr.Slider(
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=64,
                                    label="Height",
                                )
                            
                            with gr.Row():
                                steps = gr.Slider(
                                    minimum=10,
                                    maximum=100,
                                    value=50,
                                    step=5,
                                    label="Inference Steps",
                                )
                                guidance = gr.Slider(
                                    minimum=1.0,
                                    maximum=15.0,
                                    value=7.5,
                                    step=0.5,
                                    label="Guidance Scale",
                                )
                            
                            seed = gr.Slider(
                                minimum=0,
                                maximum=2147483647,
                                value=0,
                                step=1,
                                label="Seed (0 for random)",
                            )
                            
                            generate_btn = gr.Button("Generate Image", variant="primary")
                        
                        with gr.Column():
                            output_image = gr.Image(label="Generated Image", type="pil")
                            save_btn = gr.Button("Save Image")
                            save_path = gr.Textbox(
                                label="Save Path",
                                visible=False,
                            )
                
                # Tab 2: Style Transfer
                with gr.TabItem("Style Transfer"):
                    with gr.Row():
                        with gr.Column():
                            content_image = gr.Image(
                                label="Content Image",
                                type="pil",
                            )
                            style_image = gr.Image(
                                label="Style Image",
                                type="pil",
                            )
                            
                            with gr.Row():
                                style_strength = gr.Slider(
                                    minimum=0.1,
                                    maximum=10.0,
                                    value=1.0,
                                    step=0.1,
                                    label="Style Strength",
                                )
                                style_steps = gr.Slider(
                                    minimum=50,
                                    maximum=500,
                                    value=300,
                                    step=50,
                                    label="Style Transfer Steps",
                                )
                            
                            style_btn = gr.Button("Apply Style Transfer", variant="primary")
                        
                        with gr.Column():
                            styled_image = gr.Image(label="Styled Image", type="pil")
                            save_style_btn = gr.Button("Save Image")
                            save_style_path = gr.Textbox(
                                label="Save Path",
                                visible=False,
                            )
                
                # Tab 3: Combined Generation
                with gr.TabItem("Text + Style"):
                    with gr.Row():
                        with gr.Column():
                            combined_prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="Describe the image you want to generate...",
                                lines=3,
                            )
                            combined_negative = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="What to avoid in the image...",
                                lines=2,
                            )
                            combined_style = gr.Image(
                                label="Style Reference",
                                type="pil",
                            )
                            
                            with gr.Accordion("Advanced Settings", open=False):
                                with gr.Row():
                                    combined_width = gr.Slider(
                                        minimum=256,
                                        maximum=1024,
                                        value=512,
                                        step=64,
                                        label="Width",
                                    )
                                    combined_height = gr.Slider(
                                        minimum=256,
                                        maximum=1024,
                                        value=512,
                                        step=64,
                                        label="Height",
                                    )
                                
                                with gr.Row():
                                    combined_steps = gr.Slider(
                                        minimum=10,
                                        maximum=100,
                                        value=50,
                                        step=5,
                                        label="Generation Steps",
                                    )
                                    combined_guidance = gr.Slider(
                                        minimum=1.0,
                                        maximum=15.0,
                                        value=7.5,
                                        step=0.5,
                                        label="Guidance Scale",
                                    )
                                
                                combined_seed = gr.Slider(
                                    minimum=0,
                                    maximum=2147483647,
                                    value=0,
                                    step=1,
                                    label="Seed (0 for random)",
                                )
                                
                                with gr.Row():
                                    combined_style_strength = gr.Slider(
                                        minimum=0.1,
                                        maximum=10.0,
                                        value=1.0,
                                        step=0.1,
                                        label="Style Strength",
                                    )
                                    combined_style_steps = gr.Slider(
                                        minimum=50,
                                        maximum=500,
                                        value=300,
                                        step=50,
                                        label="Style Transfer Steps",
                                    )
                            
                            combined_btn = gr.Button("Generate Styled Image", variant="primary")
                        
                        with gr.Column():
                            combined_image = gr.Image(label="Generated Styled Image", type="pil")
                            save_combined_btn = gr.Button("Save Image")
                            save_combined_path = gr.Textbox(
                                label="Save Path",
                                visible=False,
                            )
            
            # Set up event handlers
            generate_btn.click(
                fn=self._generate_from_text,
                inputs=[
                    text_prompt,
                    negative_prompt,
                    width,
                    height,
                    steps,
                    guidance,
                    seed,
                ],
                outputs=output_image,
            )
            
            style_btn.click(
                fn=self._apply_style_transfer,
                inputs=[
                    content_image,
                    style_image,
                    style_strength,
                    style_steps,
                ],
                outputs=styled_image,
            )
            
            combined_btn.click(
                fn=self._combined_generation,
                inputs=[
                    combined_prompt,
                    combined_negative,
                    combined_style,
                    combined_width,
                    combined_height,
                    combined_steps,
                    combined_guidance,
                    combined_seed,
                    combined_style_strength,
                    combined_style_steps,
                ],
                outputs=combined_image,
            )
            
            # Save buttons
            save_btn.click(
                fn=self._save_image,
                inputs=output_image,
                outputs=save_path,
            )
            
            save_style_btn.click(
                fn=self._save_image,
                inputs=styled_image,
                outputs=save_style_path,
            )
            
            save_combined_btn.click(
                fn=self._save_image,
                inputs=combined_image,
                outputs=save_combined_path,
            )
        
        return interface


def launch_ui(port: int = 7860, share: bool = False):
    """Launch the Gradio UI."""
    ui = ArtGeneratorUI()
    interface = ui.build_interface()
    interface.launch(server_port=port, share=share)