import sqlite3
from typing import Dict, Optional, Union

# Create an in-memory SQLite database and seed it with sample data.
# In a real application, you would connect to a persistent database file or service.
_CONN = sqlite3.connect(":memory:", check_same_thread=False)
_CONN.row_factory = sqlite3.Row

def _initialize_db() -> None:
    with _CONN:
        _CONN.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                phone TEXT,
                email TEXT
            )
            """
        )
        # Seed only if the table is empty
        count = _CONN.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
        if count == 0:
            _CONN.executemany(
                "INSERT INTO users (user_id, name, phone, email) VALUES (?, ?, ?, ?)",
                [
                    ("user_1", "Alice Johnson", "+1-555-0100", "alice@example.com"),
                    ("user_2", "Bob Smith", "+1-555-0101", "bob@example.com"),
                    ("user_3", "Carla Gomez", "+1-555-0102", "carla@example.com"),
                ],
            )

_initialize_db()

def get_user_info(user_id: str, new_email: Optional[str] = None) -> Union[Optional[Dict[str, Optional[str]]], bool]:
    """
    Retrieve user information or update the user's email address.

    When new_email is None:
        - Returns a dictionary with keys: 'user_id', 'name', 'phone', 'email' if the user exists,
          otherwise None.

    When new_email is provided:
        - Updates the user's email and returns True if the update affected a row,
          otherwise returns False.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")

    if new_email is None:
        cur = _CONN.execute(
            "SELECT user_id, name, phone, email FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None

        return {
            "user_id": row["user_id"],
            "name": row["name"],
            "phone": row["phone"],
            "email": row["email"],
        }

    if not isinstance(new_email, str) or not new_email.strip():
        raise ValueError("new_email must be a non-empty string")

    with _CONN:
        cur = _CONN.execute(
            "UPDATE users SET email = ? WHERE user_id = ?",
            (new_email, user_id),
        )
        return cur.rowcount > 0

__all__ = ["get_user_info"]
