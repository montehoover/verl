import os
import sqlite3


_DB_PATH_ENV = "DB_PATH"
_DEFAULT_DB_PATH = "app.db"
_CSRF_TOKEN_ENV = "CSRF_TOKEN"
_CSRF_HEADER_NAME = "X-CSRF-Token"


def _get_db_connection() -> sqlite3.Connection:
    """
    Returns a sqlite3 connection to the application's database.

    The database path is read from environment variable DB_PATH, falling back
    to 'app.db' in the current working directory if not set.
    """
    db_path = os.getenv(_DB_PATH_ENV, _DEFAULT_DB_PATH)
    return sqlite3.connect(db_path)


def _get_expected_csrf_token() -> str | None:
    """
    Returns the expected CSRF token from environment variable CSRF_TOKEN.
    If not set, returns None.
    """
    return os.getenv(_CSRF_TOKEN_ENV)


def _extract_csrf_token(headers: dict) -> str | None:
    """
    Extracts CSRF token from headers in a case-insensitive manner.
    Looks for the header name defined by _CSRF_HEADER_NAME.
    """
    target = _CSRF_HEADER_NAME.lower()
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == target:
            return v if isinstance(v, str) else None
    return None


def check_user_existence(user_id: str, headers: dict) -> bool:
    """
    Verify if a user exists in the system's database and validate the CSRF token.

    Args:
        user_id: The unique identifier of the user (string).
        headers: A dictionary of request headers that must include a CSRF token
                 under the key 'X-CSRF-Token' (case-insensitive).

    Returns:
        True only if the CSRF token matches the predefined expected value AND the user exists.
        False otherwise.

    Notes:
        - Expects a 'users' table with a primary key or unique column 'id' (TEXT).
        - The database path can be set with the DB_PATH environment variable.
        - The expected CSRF token must be provided via the CSRF_TOKEN environment variable.
    """
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a str")
    if not isinstance(headers, dict):
        raise TypeError("headers must be a dict")

    if user_id == "":
        return False

    # Validate CSRF token first
    expected_token = _get_expected_csrf_token()
    if not expected_token:
        return False  # Cannot validate without an expected token configured

    provided_token = _extract_csrf_token(headers)
    if not provided_token or provided_token != expected_token:
        return False

    # Only check the database if CSRF is valid
    try:
        conn = _get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM users WHERE id = ? LIMIT 1", (user_id,))
            return cur.fetchone() is not None
        finally:
            conn.close()
    except sqlite3.Error:
        # If the DB is misconfigured or inaccessible, conservatively return False.
        return False
