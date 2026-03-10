import os
import sqlite3
from typing import Optional, Dict, Any


def get_user_info(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user's information from the database by user ID.

    Args:
        user_id (str): The user's unique identifier.

    Returns:
        dict: A dictionary of the user's details if found.
        None: If the user is not found.
    """
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a string")

    db_path = os.getenv("DATABASE_PATH", "app.db")

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()
