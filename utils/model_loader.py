import joblib

from config import config

_cache = {"model": None, "mtime": None}


def get_model():
    """Retourne le modele en cache, ou le charge/recharge si necessaire."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Aucun modele trouve a {config.MODEL_PATH}. "
            "Lancez le script d'entrainement avant de demarrer l'API."
        )

    current_mtime = config.MODEL_PATH.stat().st_mtime

    if _cache["model"] is None or _cache["mtime"] != current_mtime:
        _cache["model"] = joblib.load(config.MODEL_PATH)
        _cache["mtime"] = current_mtime

    return _cache["model"]


def predict_score(text):
    """Calcule le score de sentiment d'un texte : P(positive) - P(negative)."""
    bundle = get_model()
    vector = bundle["vectorizer"].transform([text])
    p_positive = bundle["positive_model"].predict_proba(vector)[0][1]
    p_negative = bundle["negative_model"].predict_proba(vector)[0][1]
    return round(float(p_positive - p_negative), 4)