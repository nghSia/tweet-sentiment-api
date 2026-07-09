from flask import jsonify


class APIError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class EmptyTweetListError(APIError):
    def __init__(self):
        super().__init__("La liste de tweets ne peut pas etre vide.", 400)


class InvalidFormatError(APIError):
    def __init__(self, detail="Le format attendu est un tableau JSON de chaines de caracteres."):
        super().__init__(detail, 400)


class DatabaseUnavailableError(APIError):
    def __init__(self):
        super().__init__("La base de donnees est inaccessible.", 503)


class ModelUnavailableError(APIError):
    def __init__(self, detail="Le modele de sentiment n'est pas disponible."):
        super().__init__(detail, 503)


def register_error_handlers(app):
    @app.errorhandler(APIError)
    def handle_api_error(error):
        response = jsonify({"error": error.message})
        response.status_code = error.status_code
        return response

    @app.errorhandler(404)
    def handle_not_found(error):
        return jsonify({"error": "Endpoint introuvable."}), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        return jsonify({"error": "Methode HTTP non autorisee pour cet endpoint."}), 405

    @app.errorhandler(500)
    def handle_internal_error(error):
        return jsonify({"error": "Erreur interne du serveur."}), 500