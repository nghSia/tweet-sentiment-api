import os
import tempfile

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _new_vectorizer():
    # n-grammes de caracteres : robustes au vocabulaire inedit et aux
    # fautes de frappe ; accuracy validation 0.35 -> 0.55 a donnees constantes.
    return TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
    )


def _new_classifier():
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )


class SentimentModel:
    def __init__(self):
        self.vectorizer = _new_vectorizer()
        self.positive_model = _new_classifier()
        self.negative_model = _new_classifier()

    def fit(self, texts, y_positive, y_negative):
        X = self.vectorizer.fit_transform(texts)
        self.positive_model.fit(X, y_positive)
        self.negative_model.fit(X, y_negative)
        return self

    def predict_scores(self, texts):
        X = self.vectorizer.transform(texts)
        p_positive = self.positive_model.predict_proba(X)[:, 1]
        p_negative = self.negative_model.predict_proba(X)[:, 1]
        return p_positive - p_negative

    def predict_labels(self, texts):
        X = self.vectorizer.transform(texts)
        return self.positive_model.predict(X), self.negative_model.predict(X)

    def to_bundle(self):
        return {
            "vectorizer": self.vectorizer,
            "positive_model": self.positive_model,
            "negative_model": self.negative_model,
        }

    def save(self, path):
        path = str(path)
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".pkl.tmp")
        os.close(fd)
        try:
            joblib.dump(self.to_bundle(), tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @classmethod
    def load(cls, path):
        bundle = joblib.load(path)
        model = cls()
        model.vectorizer = bundle["vectorizer"]
        model.positive_model = bundle["positive_model"]
        model.negative_model = bundle["negative_model"]
        return model
