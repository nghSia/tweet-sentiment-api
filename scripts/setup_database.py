import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mysql.connector import Error

from db.connection import get_connection
from config import config


def setup_database():
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {config.MYSQL_DATABASE} "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        print(f"Base '{config.MYSQL_DATABASE}' prete.")

        cursor.execute(f"USE {config.MYSQL_DATABASE}")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tweets (
                id INT AUTO_INCREMENT PRIMARY KEY,
                text TEXT NOT NULL,
                positive TINYINT(1) NOT NULL DEFAULT 0,
                negative TINYINT(1) NOT NULL DEFAULT 0
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        connection.commit()
        print("Table 'tweets' prete.")

    except Error as e:
        print(f"Erreur lors de l'initialisation de la base : {e}")
        sys.exit(1)
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


if __name__ == "__main__":
    setup_database()