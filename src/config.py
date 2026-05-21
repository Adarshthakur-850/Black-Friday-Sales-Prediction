import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
TRAIN_DATA_PATH = DATA_DIR / "black_friday.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
