import logging

from flask import Flask

from api.errors import register_error_handlers
from api.routes import bp as api_bp
from config import config
from utils.paths import LOGS_DIR


def create_app():
    app = Flask(__name__)

    logging.basicConfig(
        filename=LOGS_DIR / "api.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    app.register_blueprint(api_bp)
    register_error_handlers(app)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=config.DEBUG, port=config.PORT, use_reloader=False)