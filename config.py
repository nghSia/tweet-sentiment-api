import os

from dotenv import load_dotenv

from utils.path import BASE_DIR

load_dotenv(BASE_DIR / ".env")


class Config:
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "sentiment_db")

    DEBUG = os.getenv("FLASK_DEBUG", "True") == "True"
    PORT = int(os.getenv("FLASK_PORT", 5000))

    MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "models/sentiment_model.pkl")


config = Config()