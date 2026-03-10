import os
import re
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
    Verify if a user exists in the database and validate the user's phone number format.

    Args:
        user_id (str): The unique identifier of the user to check.

    Returns:
        bool: True if the user exists, False otherwise.

    Notes:
        - Expects a SQLite database located at USER_DB_PATH (env var) or 'app.db' by default.
        - Expects a 'users' table to exist.
        - Will look for the user identifier in either an 'id' or 'user_id' column.
        - If a phone number column is found, validates it against the pattern XXX-XXX-XXXX.
          If invalid (including missing/empty), a message is printed.
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

            # Determine which column to use for user lookup ('id' preferred, fallback to 'user_id')
            cur.execute("PRAGMA table_info(users)")
            columns = [row[1] for row in cur.fetchall()]  # row[1] is the column name
            lookup_col = "id" if "id" in columns else ("user_id" if "user_id" in columns else None)
            if lookup_col is None:
                return False

            # Determine phone column if present
            phone_candidates = ["phone", "phone_number", "phone_no", "phoneNo", "mobile", "mobile_number"]
            phone_col = next((c for c in phone_candidates if c in columns), None)

            # Query for existence (and phone if available)
            if phone_col:
                cur.execute(f"SELECT {phone_col} FROM users WHERE {lookup_col} = ? LIMIT 1", (uid,))
                row = cur.fetchone()
                if row is None:
                    return False  # user not found
                phone_value = row[0]
                pattern = re.compile(r"^\d{3}-\d{3}-\d{4}$")
                if not (isinstance(phone_value, str) and pattern.match(phone_value.strip())):
                    print(f"Invalid phone number format for user {uid}: expected XXX-XXX-XXXX.")
                return True
            else:
                # Fallback: we can still verify existence, but we can't read a phone number
                cur.execute(f"SELECT 1 FROM users WHERE {lookup_col} = ? LIMIT 1", (uid,))
                exists = cur.fetchone() is not None
                if exists:
                    print(f"Invalid phone number format for user {uid}: expected XXX-XXX-XXXX.")
                return exists
    except sqlite3.Error:
        # If any database error occurs, treat as not found (False)
        return False
