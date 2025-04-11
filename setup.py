"""
Setup script for the AI Art Generator package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="artor",
    version="0.1.0",
    author="culusTech",
    author_email="calculus069@gmail.com",
    description="An AI-powered art generator using Stable Diffusion and style transfer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Odeneho-Calculus/artor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.20.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "ftfy>=6.1.1",
        "scipy>=1.10.1",
        "Pillow>=9.5.0",
        "gradio>=3.40.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "artor=main:main",
        ],
    },
)