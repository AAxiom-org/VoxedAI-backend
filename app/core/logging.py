import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# Path to the logs directory
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "voxai.log"

# Create logs directory if it doesn't exist
LOG_DIR.mkdir(exist_ok=True)


class InterceptHandler(logging.Handler):
    """
    Intercepts standard logging messages and redirects them to loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    """
    Configures logging for the application.
    """
    # Remove all existing handlers
    logging.root.handlers = []
    
    # Configure loguru to split logging between console and file
    logger.configure(
        handlers=[
            # Console handler with INFO level (colorized)
            {"sink": sys.stderr, "level": "INFO", "colorize": True},
            # File handler with DEBUG level
            {"sink": LOG_FILE, "level": "DEBUG", "rotation": "20 MB", "retention": "1 week"},
        ],
        levels=[
            {"name": "DEBUG", "color": "<cyan>"},
            {"name": "INFO", "color": "<green>"},
            {"name": "WARNING", "color": "<yellow>"},
            {"name": "ERROR", "color": "<red>"},
            {"name": "CRITICAL", "color": "<red><bold>"},
        ],
    )

    # Intercept standard logging messages
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)

    # Disable logging for some noisy libraries
    for logger_name in ["uvicorn.access"]:
        logging.getLogger(logger_name).handlers = []
    
    # Set level of specific loggers
    for logger_name in [
        "uvicorn",
        "uvicorn.error",
        "fastapi",
    ]:
        logging.getLogger(logger_name).handlers = [InterceptHandler()]
        
    return logger 