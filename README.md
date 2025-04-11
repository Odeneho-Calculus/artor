# 🎨 ARTOR

A powerful AI-powered tool that combines Stable Diffusion for text-to-image generation with neural style transfer capabilities to create stunning, customized artwork.

## ✨ Features

- **Text-to-Image Generation**: Create images from text descriptions using state-of-the-art Stable Diffusion models
- **Style Transfer**: Apply the artistic style of one image to another using neural style transfer
- **Combined Workflow**: Generate images from text prompts and automatically apply artistic styles
- **User-Friendly Interface**: Simple Gradio-based UI for easy interaction
- **Advanced Controls**: Fine-tune generation parameters for perfect results

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) for faster generation

### Setting up the environment

```bash
# Clone the repository
git clone https://github.com/Odeneho-Calculus/artor.git
cd artor

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## 🖥️ Usage

### Starting the application

```bash
# Run the application with default settings
python main.py

# Run with specific port
python main.py --port 8080

# Run with public sharing enabled
python main.py --share
```

Once started, open your web browser and navigate to http://localhost:7860 (or the port you specified).

### Using the interface

The application provides three main features:

1. **Text to Image**: Enter a text prompt to generate an image
   - Use the negative prompt to specify what you don't want to see
   - Adjust width, height, steps, and guidance for fine control
   - Set a seed value for reproducible results

2. **Style Transfer**: Apply artistic styles to your images
   - Upload a content image (what you want to style)
   - Upload a style image (the artistic style to apply)
   - Adjust style strength and steps for different effects

3. **Text + Style**: Generate an image from text and apply a style in one go
   - Combines both processes with a single button click
   - All controls from both processes are available

## 🔧 Advanced Configuration

You can modify default settings in `config/settings.py`:

- Model selection and parameters
- Default image sizes
- Performance settings
- Style transfer weights

## 🧪 Testing

Run the tests to ensure everything is working correctly:

```bash
pytest tests/
```

## 📚 Project Structure

```
ai-art-generator/
│
├── app/                      # Application core
│   ├── generator/            # Image generation modules
│   ├── utils/                # Utility functions
│   └── interface/            # User interface components
│
├── config/                   # Configuration files
├── models/                   # Directory to store model weights
├── examples/                 # Example images and prompts
├── tests/                    # Unit and integration tests
├── main.py                   # Entry point
└── README.md                 # Project documentation
```

## 👨‍💻 Development

### Adding new features

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Implement your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for the text-to-image model
- [PyTorch](https://pytorch.org/) for the neural network framework
- [Gradio](https://gradio.app/) for the user interface