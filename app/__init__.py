"""
AI Art Generator application package.
"""
import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create a logger for this package
logger = logging.getLogger(__name__)
logger.info("Initializing AI Art Generator")

# Package version
__version__ = "0.1.0"