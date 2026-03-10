import os
import sqlite3
from typing import Optional, Dict


def get_user_info(user_id: str, auth_token: str) -> Optional[Dict[str, str]]:
    """
    Retrieve user information from the database by user_id,
    only if the provided auth_token matches the predefined token.

    Args:
        user_id: The unique identifier of the user (string).
        auth_token: The authentication token to verify (string).

    Returns:
        A dictionary containing the user's email, e.g., {"email": "user@example.com"},
        or None if the user does not exist or authentication fails.
    """
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a str")
    if not isinstance(auth_token, str):
        raise TypeError("auth_token must be a str")

    expected_token = os.getenv("AUTH_TOKEN")
    if not expected_token or auth_token != expected_token:
        return None

    db_path = os.getenv("DB_PATH", "app.db")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT email FROM users WHERE id = ? LIMIT 1", (user_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return {"email": row[0]}
