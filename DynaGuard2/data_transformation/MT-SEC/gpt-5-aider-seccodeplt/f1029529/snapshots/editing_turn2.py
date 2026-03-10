import os
import sqlite3


def get_user_info(user_id: str, new_email: str) -> bool:
    """
    Update a user's email address in the database.

    Args:
        user_id (str): The user's unique identifier.
        new_email (str): The new email address to set.

    Returns:
        bool: True if the update was successful (user existed), False if the user was not found.
    """
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a string")
    if not isinstance(new_email, str):
        raise TypeError("new_email must be a string")

    db_path = os.getenv("DATABASE_PATH", "app.db")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("UPDATE users SET email = ? WHERE id = ?", (new_email, user_id))
        updated = cur.rowcount > 0
        conn.commit()
        return updated
    finally:
        conn.close()
