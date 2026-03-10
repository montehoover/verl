import os
import sqlite3
import hmac
import hashlib
import base64
from typing import Optional


def _is_valid_csrf(user_id: str, csrf_token: str) -> bool:
    """
    Validate a CSRF token for the given user_id.

    The expected token format is a base64url-encoded HMAC-SHA256 of the user_id
    using a server-side secret provided via the CSRF_SECRET environment variable.

    expected_token = base64urlencode(HMAC_SHA256(CSRF_SECRET, user_id))

    The comparison is done using hmac.compare_digest, and incoming tokens are
    normalized by stripping any trailing '=' padding to tolerate both padded
    and unpadded base64url inputs.
    """
    secret = os.getenv("CSRF_SECRET")
    if not secret or not csrf_token or not isinstance(csrf_token, str):
        return False

    mac = hmac.new(secret.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256).digest()
    expected = base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")
    provided = csrf_token.strip().rstrip("=")
    return hmac.compare_digest(expected, provided)


def get_user_info(user_id: str, csrf_token: str) -> Optional[str]:
    """
    Retrieve a user's email from the database by user ID only if the CSRF token is valid.

    Environment:
      - USER_DB_PATH: Path to a SQLite database file containing a table named 'users'
        with columns:
          - id (TEXT PRIMARY KEY)
          - email (TEXT)
      - CSRF_SECRET: Secret string used to validate CSRF tokens.

    CSRF validation:
      The CSRF token must be base64url(HMAC_SHA256(CSRF_SECRET, user_id)).

    Args:
        user_id: The ID of the user as a string.
        csrf_token: The CSRF token associated with the request.

    Returns:
        The user's email as a string if found and CSRF token is valid, otherwise None.
    """
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a string")
    if not isinstance(csrf_token, str):
        raise TypeError("csrf_token must be a string")

    # Validate CSRF token before any database access
    if not _is_valid_csrf(user_id, csrf_token):
        return None

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
