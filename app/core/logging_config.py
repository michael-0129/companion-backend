import logging
import sys
from app.core.config import settings

# Determine log level from settings, default to INFO
LOG_LEVEL = settings.LOG_LEVEL.upper() if hasattr(settings, 'LOG_LEVEL') else "INFO"
numeric_level = getattr(logging, LOG_LEVEL, logging.INFO)

# Basic configuration
logging.basicConfig(
    level=numeric_level,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout) # Output to console
    ]
)

def get_logger(name: str):
    """
    Retrieves a logger instance.
    """
    return logging.getLogger(name)

# Example of how to get a logger in other modules:
# from app.core.logging_config import get_logger
# logger = get_logger(__name__)
# logger.info("This is an info message.") 