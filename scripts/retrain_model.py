import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config
from ml.training import train_and_save
from utils.paths import LOGS_DIR, MODEL_HISTORY_DIR

logging.basicConfig(
    filename=LOGS_DIR / "retrain.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def retrain():
    logger.info("Debut du reentrainement.")
    try:
        metrics = train_and_save()
    except Exception:
        logger.exception("Echec du reentrainement, l'ancien modele reste en place.")
        raise

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = MODEL_HISTORY_DIR / f"sentiment_model_{timestamp}.pkl"
    shutil.copy2(config.MODEL_PATH, history_path)

    logger.info(
        "Reentrainement reussi. dataset=%s train=%s validation=%s "
        "f1_positive=%s f1_negative=%s archive=%s",
        metrics["dataset_size"],
        metrics["train_size"],
        metrics["validation_size"],
        metrics["f1_positive"],
        metrics["f1_negative"],
        history_path.name,
    )
    return metrics


if __name__ == "__main__":
    try:
        results = retrain()
    except Exception as exc:
        print(f"Echec : {exc}")
        sys.exit(1)
    print("Reentrainement termine.")
    for key, value in results.items():
        print(f"  {key}: {value}")
