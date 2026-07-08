import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db.connection import get_connection
from db.queries import clear_tweets, insert_many_from_dicts


CSV_PATH = Path(__file__).resolve().parent / "data" / "tweets_sample.csv"


def load_csv_rows(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV introuvable: {CSV_PATH}")

    rows = load_csv_rows(CSV_PATH)
    connection = get_connection()
    try:
        clear_tweets(connection)
        inserted = insert_many_from_dicts(connection, rows)
        print(f"{inserted} tweets charges depuis {CSV_PATH.name}.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()