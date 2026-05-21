import logging
import sys
from pathlib import Path
import joblib
from .config import LOGS_DIR, LOG_LEVEL

def setup_logger(name: str):
    """
    Sets up a logger with the specified name and log level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Prevent adding multiple handlers if logger is already configured
    if logger.handlers:
        return logger

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(LOG_LEVEL)
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # File Handler
    f_handler = logging.FileHandler(LOGS_DIR / "app.log")
    f_handler.setLevel(LOG_LEVEL)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger

logger = setup_logger("utils")

def save_object(obj, file_path: Path):
    """
    Saves a Python object to the specified file path using joblib.
    """
    try:
        joblib.dump(obj, file_path)
        logger.info(f"Object saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving object to {file_path}: {e}")
        raise

def load_object(file_path: Path):
    """
    Loads a Python object from the specified file path using joblib.
    """
    try:
        obj = joblib.load(file_path)
        logger.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {file_path}: {e}")
        raise
