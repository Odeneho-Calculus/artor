"""
Neural style transfer implementation using PyTorch.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from config.settings import (
    CONTENT_WEIGHT,
    DEVICE,
    NUM_STYLE_TRANSFER_STEPS,
    STYLE_WEIGHT,
)

logger = logging.getLogger(__name__)


class ContentLoss(nn.Module):
    """Content loss module for neural style transfer."""
    
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        
    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    """Style loss module for neural style transfer."""

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self._gram_matrix(target_feature).detach()
        
    def forward(self, x):
        G = self._gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x
    
    @staticmethod
    def _gram_matrix(x):
        batch_size, n_channels, height, width = x.size()
        features = x.view(batch_size * n_channels, height * width)
        G = torch.mm(features, features.t())
        return G.div(batch_size * n_channels * height * width)


class StyleTransfer:
    """Neural style transfer implementation."""
    
    def __init__(self):
        """Initialize the style transfer module."""
        self.device = torch.device(DEVICE)
        logger.info(f"Style transfer using device: {self.device}")
        
        # Load VGG19 model
        try:
            self.cnn = models.vgg19(weights="DEFAULT").features.to(self.device).eval()
        except Exception as e:
            logger.warning(f"Error loading VGG19 with cuda: {e}")
            logger.info("Falling back to CPU for style transfer")
            self.device = torch.device("cpu")
            self.cnn = models.vgg19(weights="DEFAULT").features.to(self.device).eval()
            
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        
        # Content and style layers
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        logger.info("Style transfer module initialized")
    
    def _load_image(self, image_path: str, size: int = None) -> torch.Tensor:
        """Load an image and convert to tensor."""
        image = Image.open(image_path)
        
        # Handle size if provided
        if size is not None:
            image = transforms.Resize(size)(image)
        
        # Convert to tensor and normalize
        loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cnn_normalization_mean, 
                               std=self.cnn_normalization_std)
        ])
        
        image = loader(image).unsqueeze(0).to(self.device)
        return image
    
    def _prepare_model_and_losses(self, content_img, style_img):
        """Prepare the model with content and style losses."""
        cnn = self.cnn
        content_losses = []
        style_losses = []
        
        model = nn.Sequential()
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
                
            model.add_module(name, layer)
            
            # Add content losses
            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)
                
            # Add style losses
            if name in self.style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)
                
        # Trim the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]
        
        return model, style_losses, content_losses
    
    def _tensor_to_image(self, tensor):
        """Convert a tensor to a PIL image."""
        tensor = tensor.clone().detach().cpu()
        tensor = tensor * torch.tensor(self.cnn_normalization_std).view(-1, 1, 1) + \
                 torch.tensor(self.cnn_normalization_mean).view(-1, 1, 1)
        tensor = tensor.clamp(0, 1)
        
        transform = transforms.ToPILImage()
        return transform(tensor)
    
    def transfer_style(
        self,
        content_img: Image.Image,
        style_img: Image.Image,
        num_steps: int = NUM_STYLE_TRANSFER_STEPS,
        style_weight: float = STYLE_WEIGHT,
        content_weight: float = CONTENT_WEIGHT,
        size: int = 512,
    ) -> Image.Image:
        """
        Apply style transfer to combine content and style images.
        
        Args:
            content_img: Content image (PIL Image)
            style_img: Style image (PIL Image)
            num_steps: Number of optimization steps
            style_weight: Weight for style loss
            content_weight: Weight for content loss
            size: Size to resize images to before processing
            
        Returns:
            Stylized image as PIL Image
        """
        logger.info("Starting style transfer process")
        
        # Convert PIL images to tensors
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cnn_normalization_mean, 
                               std=self.cnn_normalization_std)
        ])
        
        content_tensor = transform(content_img).unsqueeze(0).to(self.device)
        style_tensor = transform(style_img).unsqueeze(0).to(self.device)
        
        # Create a tensor from the content image to optimize
        input_tensor = content_tensor.clone()
        
        # Prepare model and losses
        model, style_losses, content_losses = self._prepare_model_and_losses(
            content_tensor, style_tensor
        )
        
        # Setup optimizer
        optimizer = optim.LBFGS([input_tensor.requires_grad_()])
        
        logger.info(f"Running optimization for {num_steps} steps")
        
        step = 0
        
        def closure():
            nonlocal step
            input_tensor.data.clamp_(0, 1)
            
            optimizer.zero_grad()
            model(input_tensor)
            
            style_score = 0
            content_score = 0
            
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
                
            style_score *= style_weight
            content_score *= content_weight
            
            loss = style_score + content_score
            loss.backward()
            
            step += 1
            if step % 50 == 0:
                logger.info(f"Step {step}/{num_steps}")
                
            return loss
        
        # Run the optimization
        for _ in range(num_steps):
            optimizer.step(closure)
        
        # Clamp the result and convert back to image
        input_tensor.data.clamp_(0, 1)
        result_image = self._tensor_to_image(input_tensor[0])
        
        logger.info("Style transfer completed")
        return result_image