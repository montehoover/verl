import os
import sqlite3
from contextlib import closing


__all__ = ["check_user_exists"]


def _get_db_path() -> str:
    """
    Returns the SQLite database file path.
    Uses the USER_DB_PATH environment variable if set, otherwise defaults to 'app.db'.
    """
    return os.getenv("USER_DB_PATH", "app.db")


def _connect() -> sqlite3.Connection:
    """Create and return a SQLite connection to the user database."""
    return sqlite3.connect(_get_db_path())


def check_user_exists(user_id: str) -> bool:
    """
    Verify if a user exists in the database.

    Args:
        user_id (str): The unique identifier of the user to check.

    Returns:
        bool: True if the user exists, False otherwise.

    Notes:
        - Expects a SQLite database located at USER_DB_PATH (env var) or 'app.db' by default.
        - Expects a 'users' table to exist.
        - Will look for the user identifier in either an 'id' or 'user_id' column.
    """
    if not isinstance(user_id, str):
        return False

    uid = user_id.strip()
    if uid == "":
        return False

    try:
        with _connect() as conn, closing(conn.cursor()) as cur:
            # Ensure 'users' table exists
            cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND lower(name)='users'")
            if cur.fetchone() is None:
                return False

            # Determine which column to use ('id' preferred, fallback to 'user_id')
            cur.execute("PRAGMA table_info(users)")
            columns = [row[1] for row in cur.fetchall()]  # row[1] is the column name
            lookup_col = "id" if "id" in columns else ("user_id" if "user_id" in columns else None)
            if lookup_col is None:
                return False

            query = f"SELECT 1 FROM users WHERE {lookup_col} = ? LIMIT 1"
            cur.execute(query, (uid,))
            return cur.fetchone() is not None
    except sqlite3.Error:
        # If any database error occurs, treat as not found (False)
        return False
