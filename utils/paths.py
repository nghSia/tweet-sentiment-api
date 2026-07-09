from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
MODEL_HISTORY_DIR = MODELS_DIR / "model_history"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "scripts" / "data"

for directory in (MODELS_DIR, MODEL_HISTORY_DIR, LOGS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)