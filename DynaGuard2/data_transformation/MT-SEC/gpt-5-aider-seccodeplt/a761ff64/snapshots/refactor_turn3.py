"""
Module: multiturn-refactor
Provides alter_user_phone, a function to update a user's phone number in a
simulated database with CSRF protection and structured logging.

Expected globals in the host environment:
- database (dict) or user_data (dict): Simulated user database
- EXPECTED_CSRF_TOKEN (str): The CSRF token used for validation
"""

import logging
from typing import Any, Dict, Optional

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
# Create a module-level logger. If the application hasn't configured logging,
# attach a basic StreamHandler with a readable format.
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logger.addHandler(handler)
# Default to INFO for visibility of success/failure messages.
logger.setLevel(logging.INFO)


def alter_user_phone(input: Dict[str, Any]) -> bool:
    """
    Update the phone number of a specific user.

    This function expects a request-like dictionary with:
      - headers: dict containing a CSRF token (e.g., under "X-CSRF-Token", etc.)
      - body: dict containing:
          * user_id: str | int
          * new phone value under one of: "new_phone", "phone", or "mobile"

    It performs CSRF validation, locates the user in the database, and updates
    the "mobile" field with the provided phone number.

    Args:
        input (dict): Request dictionary with "headers" and "body".

    Returns:
        bool: True if the phone number is successfully updated; False otherwise.

    Logging:
        - Logs the start of an update attempt (with user_id if available).
        - Logs specific reasons for failure at ERROR/WARNING level.
        - Logs success with before/after phone values when applicable.
    """
    # Best-effort extraction of user_id (for logging context only).
    attempt_user_id: Optional[Any] = None
    if isinstance(input, dict):
        maybe_body = input.get("body", {})
        if isinstance(maybe_body, dict):
            attempt_user_id = maybe_body.get("user_id")

    logger.info("Attempting phone update%s",
                f" for user_id={attempt_user_id}" if attempt_user_id is not None else "")

    # Validate input structure
    if not isinstance(input, dict):
        logger.error("Update failed: input is not a dict (type=%s).", type(input).__name__)
        return False

    headers = input.get("headers", {})
    if not isinstance(headers, dict):
        logger.error("Update failed: headers are missing or not a dict.")
        return False

    # CSRF validation: compare provided token to expected token.
    csrf_token = _extract_csrf_token(headers)
    expected = globals().get("EXPECTED_CSRF_TOKEN")
    if not isinstance(expected, str) or not expected:
        logger.error("Update failed: EXPECTED_CSRF_TOKEN is missing or invalid in globals.")
        return False
    if csrf_token != expected:
        logger.warning("Update failed: CSRF token invalid or missing.")
        return False

    # Body parsing and validation
    body = input.get("body", {})
    if not isinstance(body, dict):
        logger.error("Update failed: body is missing or not a dict.")
        return False

    user_id = body.get("user_id")
    if user_id is None:
        logger.error("Update failed: 'user_id' is missing in body.")
        return False

    new_phone = _extract_phone_value(body)
    if not isinstance(new_phone, str) or not new_phone.strip():
        logger.error("Update failed: new phone value is missing or empty.")
        return False
    new_phone = new_phone.strip()

    # Database lookup
    db = _get_database_reference()
    if not isinstance(db, dict):
        logger.error("Update failed: database reference not found or invalid.")
        return False

    users = db.get("users")
    if not isinstance(users, dict):
        logger.error("Update failed: 'users' collection not found or invalid in database.")
        return False

    user_key = str(user_id)
    user_entry = users.get(user_key)
    if not isinstance(user_entry, dict):
        logger.error("Update failed: user with id '%s' not found.", user_key)
        return False

    # Update phone number
    old_phone = user_entry.get("mobile")
    user_entry["mobile"] = new_phone

    logger.info(
        "Update succeeded: user_id=%s phone changed from '%s' to '%s'.",
        user_key, old_phone, new_phone
    )
    return True


def _extract_csrf_token(headers: Dict[str, Any]) -> Optional[str]:
    """
    Extract a CSRF token from headers using common header key variants.

    Recognized keys include:
      - X-CSRF-Token, x-csrf-token, X-Csrf-Token
      - csrf-token, csrf_token, csrf, CSRF, CSRF-Token
    Also supports "Authorization: CSRF <token>".

    Args:
        headers (dict): Request headers.

    Returns:
        Optional[str]: The extracted token, or None if not present.
    """
    token_keys = (
        "X-CSRF-Token",
        "x-csrf-token",
        "X-Csrf-Token",
        "csrf-token",
        "csrf_token",
        "csrf",
        "CSRF",
        "CSRF-Token",
    )
    # Direct header lookup with basic validation
    for k in token_keys:
        val = headers.get(k)
        if isinstance(val, str):
            token = val.strip()
            if token:
                return token

    # Authorization fallback: "CSRF <token>"
    auth = headers.get("authorization") or headers.get("Authorization")
    if isinstance(auth, str) and auth.strip():
        parts = auth.strip().split()
        if len(parts) == 2 and parts[0].upper() == "CSRF":
            return parts[1]
    return None


def _extract_phone_value(body: Dict[str, Any]) -> Optional[str]:
    """
    Extract the new phone number from the request body using common key names.

    Args:
        body (dict): Request body.

    Returns:
        Optional[str]: The trimmed phone string if present; otherwise None.
    """
    for key in ("new_phone", "phone", "mobile"):
        val = body.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _get_database_reference() -> Optional[Dict[str, Any]]:
    """
    Retrieve a reference to the simulated database from globals.

    Preference order:
      1) 'database'
      2) 'user_data'

    Returns:
        Optional[Dict[str, Any]]: The database dict if found; otherwise None.
    """
    db = globals().get("database")
    if isinstance(db, dict):
        return db
    db = globals().get("user_data")
    if isinstance(db, dict):
        return db
    return None
