from flask import Blueprint, jsonify, request

from api.errors import EmptyTweetListError, InvalidFormatError, ModelUnavailableError
from utils.model_loader import predict_score

bp = Blueprint("api", __name__)


@bp.route("/analyze", methods=["POST"])
def analyze():
    tweets = request.get_json(silent=True)

    if not isinstance(tweets, list):
        raise InvalidFormatError("Le corps de la requete doit etre un tableau JSON de chaines de caracteres.")

    if len(tweets) == 0:
        raise EmptyTweetListError()

    if not all(isinstance(t, str) for t in tweets):
        raise InvalidFormatError("Chaque element du tableau doit etre une chaine de caracteres.")

    try:
        results = {tweet: predict_score(tweet) for tweet in tweets}
    except FileNotFoundError:
        raise ModelUnavailableError()

    return jsonify(results)