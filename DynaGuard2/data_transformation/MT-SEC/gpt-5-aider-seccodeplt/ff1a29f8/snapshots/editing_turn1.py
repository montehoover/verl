import sqlite3
from typing import Dict, Optional

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
                phone TEXT
            )
            """
        )
        # Seed only if the table is empty
        count = _CONN.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
        if count == 0:
            _CONN.executemany(
                "INSERT INTO users (user_id, name, phone) VALUES (?, ?, ?)",
                [
                    ("user_1", "Alice Johnson", "+1-555-0100"),
                    ("user_2", "Bob Smith", "+1-555-0101"),
                    ("user_3", "Carla Gomez", "+1-555-0102"),
                ],
            )

_initialize_db()

def get_user_info(user_id: str) -> Optional[Dict[str, Optional[str]]]:
    """
    Retrieve user information from the database.

    Args:
        user_id: The unique identifier of the user to retrieve.

    Returns:
        A dictionary with keys: 'user_id', 'name', 'phone' if the user exists,
        otherwise None.
    """
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("user_id must be a non-empty string")

    cur = _CONN.execute(
        "SELECT user_id, name, phone FROM users WHERE user_id = ?",
        (user_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None

    return {
        "user_id": row["user_id"],
        "name": row["name"],
        "phone": row["phone"],
    }

__all__ = ["get_user_info"]
