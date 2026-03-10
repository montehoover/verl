import os
import sqlite3


_DB_PATH_ENV = "DB_PATH"
_DEFAULT_DB_PATH = "app.db"


def _get_db_connection() -> sqlite3.Connection:
    """
    Returns a sqlite3 connection to the application's database.

    The database path is read from environment variable DB_PATH, falling back
    to 'app.db' in the current working directory if not set.
    """
    db_path = os.getenv(_DB_PATH_ENV, _DEFAULT_DB_PATH)
    return sqlite3.connect(db_path)


def check_user_existence(user_id: str) -> bool:
    """
    Verify if a user exists in the system's database.

    Args:
        user_id: The unique identifier of the user (string).

    Returns:
        True if the user exists, False otherwise.

    Notes:
        Expects a 'users' table with a primary key or unique column 'id' (TEXT).
        The database path can be set with the DB_PATH environment variable.
    """
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a str")

    if user_id == "":
        return False

    try:
        conn = _get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM users WHERE id = ? LIMIT 1", (user_id,))
            return cur.fetchone() is not None
        finally:
            conn.close()
    except sqlite3.Error:
        # If the DB is misconfigured or inaccessible, conservatively return False.
        return False
