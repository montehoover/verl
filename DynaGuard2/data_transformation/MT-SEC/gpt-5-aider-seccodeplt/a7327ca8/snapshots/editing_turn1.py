import os
import sqlite3
from typing import Optional, Dict


def get_user_info(user_id: str) -> Optional[Dict[str, str]]:
    """
    Retrieve user information from the database by user_id.

    Args:
        user_id: The unique identifier of the user (string).

    Returns:
        A dictionary containing the user's email, e.g., {"email": "user@example.com"},
        or None if the user does not exist.
    """
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a str")

    db_path = os.getenv("DB_PATH", "app.db")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT email FROM users WHERE id = ? LIMIT 1", (user_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return {"email": row[0]}
