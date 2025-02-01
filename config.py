from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
DATA_PATH = RAW_DATA_DIR / 'transactions.csv'
MODEL_PATH = ARTIFACTS_DIR / 'model.joblib'
VECTORIZER_PATH = ARTIFACTS_DIR / 'vectorizer.joblib'

# Model settings
TOP_N_PREDICTIONS = 5
