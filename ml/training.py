from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import config
from db.connection import get_connection
from db.queries import fetch_all_tweets
from ml.model import SentimentModel

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
MIN_DATASET_SIZE = 20


def load_dataset():
    connection = get_connection()
    try:
        rows = fetch_all_tweets(connection)
    finally:
        connection.close()

    texts = [row["text"] for row in rows]
    y_positive = [int(row["positive"]) for row in rows]
    y_negative = [int(row["negative"]) for row in rows]
    return texts, y_positive, y_negative


def split_dataset(texts, y_positive, y_negative):
    combined = [f"{p}{n}" for p, n in zip(y_positive, y_negative)]
    try:
        return train_test_split(
            texts, y_positive, y_negative,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=combined,
        )
    except ValueError:
        return train_test_split(
            texts, y_positive, y_negative,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
        )


def train_and_save(model_path=None):
    model_path = model_path or config.MODEL_PATH

    texts, y_positive, y_negative = load_dataset()
    if len(texts) < MIN_DATASET_SIZE:
        raise RuntimeError(
            f"Dataset insuffisant : {len(texts)} tweets, minimum {MIN_DATASET_SIZE}."
        )

    (
        texts_train, texts_val,
        y_pos_train, y_pos_val,
        y_neg_train, y_neg_val,
    ) = split_dataset(texts, y_positive, y_negative)

    model = SentimentModel().fit(texts_train, y_pos_train, y_neg_train)

    pred_pos, pred_neg = model.predict_labels(texts_val)
    metrics = {
        "dataset_size": len(texts),
        "train_size": len(texts_train),
        "validation_size": len(texts_val),
        "f1_positive": round(f1_score(y_pos_val, pred_pos, zero_division=0), 4),
        "f1_negative": round(f1_score(y_neg_val, pred_neg, zero_division=0), 4),
    }

    model.save(model_path)
    return metrics


if __name__ == "__main__":
    results = train_and_save()
    print("Entrainement termine.")
    for key, value in results.items():
        print(f"  {key}: {value}")
