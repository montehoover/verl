import os
import sqlite3
from typing import Optional


def get_user_info(user_id: str) -> Optional[str]:
    """
    Retrieve a user's email from the database by user ID.

    This function expects an environment variable USER_DB_PATH that points
    to a SQLite database file containing a table named 'users' with columns:
      - id (TEXT PRIMARY KEY)
      - email (TEXT)

    Args:
        user_id: The ID of the user as a string.

    Returns:
        The user's email as a string if found, otherwise None.
    """
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a string")

    db_path = os.getenv("USER_DB_PATH")
    if not db_path:
        # No database configured; cannot look up user
        return None

    conn = None
    try:
        # Open database in read-only mode for safety
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.execute("SELECT email FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        # Any DB errors (missing file/table/etc.) result in a graceful None
        return None
    finally:
        if conn is not None:
            conn.close()
