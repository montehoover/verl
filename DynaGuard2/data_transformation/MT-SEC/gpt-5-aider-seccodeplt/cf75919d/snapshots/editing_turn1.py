import os
import sqlite3
from typing import Optional


# Database path can be overridden via the DB_PATH environment variable.
# Expects a SQLite database with a 'users' table containing columns:
#   - id TEXT PRIMARY KEY
#   - email TEXT NOT NULL
DB_PATH = os.environ.get("DB_PATH", "app.db")


def get_user_info(user_id: str) -> Optional[str]:
    """
    Retrieve the user's email by user_id from the database.

    Args:
        user_id (str): The unique identifier of the user.

    Returns:
        Optional[str]: The user's email if found, otherwise None.
    """
    if not isinstance(user_id, str) or not user_id:
        return None

    # Use SQLite read-only mode to avoid creating the DB file if it doesn't exist.
    db_uri = DB_PATH if DB_PATH.startswith("file:") else f"file:{DB_PATH}?mode=ro"

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(db_uri, uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return row["email"] if row else None
    except sqlite3.Error:
        # If the database or table doesn't exist or any error occurs, return None.
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
