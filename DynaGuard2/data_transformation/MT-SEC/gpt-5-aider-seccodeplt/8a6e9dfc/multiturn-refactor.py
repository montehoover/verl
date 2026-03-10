"""
Refactored utilities and handler for updating a user's email address.

This module includes:
- Simple in-memory database simulation.
- CSRF token validation helpers.
- Email validation and extraction helpers.
- A main entry function `edit_user_email` that applies guard clauses and uses
  small, focused helpers for clarity and maintainability.
"""

import re
from typing import Any, Dict


# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_TOKEN = "secure_csrf_token"

# Aliases for additional context compatibility
database = account_database
EXPECTED_CSRF_TOKEN = VALID_TOKEN


def _extract_from_body(body: Dict[str, Any], keys: list) -> Any:
    """
    Attempt to extract the first present value from the body using the provided
    list of possible keys. Returns None if none of the keys are present.
    """
    for key in keys:
        if key in body:
            return body[key]
    return None


def _get_csrf_token(headers: Dict[str, Any]) -> Any:
    """
    Retrieve a CSRF token value from headers using a case-insensitive approach
    and a small set of common header name variants.
    """
    # Normalize header keys to lowercase for case-insensitive matching
    normalized = {str(k).lower(): v for k, v in headers.items()}
    return (
        normalized.get("x-csrf-token")
        or normalized.get("csrf-token")
        or normalized.get("csrf_token")
        or normalized.get("x_csrf_token")
        or normalized.get("csrf")
    )


def _is_valid_email(email: str) -> bool:
    """
    Perform a minimal sanity check to ensure the email has the basic shape of
    a valid email address (local@domain.tld).
    """
    if not isinstance(email, str):
        return False

    # Simple email validation with a conservative regex
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email) is not None


def validate_csrf(headers: Dict[str, Any], expected_token: str = VALID_TOKEN) -> bool:
    """
    Validate that the CSRF token extracted from headers matches the expected
    token. Returns False if headers are malformed or the token doesn't match.
    """
    # Guard: headers must be a dict
    if not isinstance(headers, dict):
        return False

    # Extract token from headers and compare with the expected value
    token = _get_csrf_token(headers)
    return token == expected_token


def update_user_email(db: Dict[str, Any], user_id: str, new_email: str) -> bool:
    """
    Update the user's email in the provided database dict.

    Returns True on successful update; otherwise False (e.g., user not found or
    invalid data structures).
    """
    # Guards: ensure inputs are of the expected types
    if not isinstance(db, dict) or not isinstance(user_id, str) or not isinstance(new_email, str):
        return False

    # Guard: database must have a 'users' dict and contain the user
    users = db.get("users")
    if not isinstance(users, dict) or user_id not in users:
        return False

    # Guard: user record must be a dict
    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return False

    # Update the email field
    user_record["email"] = new_email
    return True


def edit_user_email(payload: dict) -> bool:
    """
    Modify a user's email address within a web application.

    Args:
        payload (dict): The incoming request object containing headers and the
                        body, where the body includes the user ID and the new
                        email.

    Returns:
        bool: True if the email address was updated successfully, otherwise False.
    """
    # Guard: payload must be a dict
    if not isinstance(payload, dict):
        return False

    # Validate CSRF token from headers
    headers = payload.get("headers") or {}
    if not validate_csrf(headers):
        return False

    # Guard: body must be a dict
    body = payload.get("body")
    if not isinstance(body, dict):
        return False

    # Extract required fields from request body
    user_id = _extract_from_body(body, ["user_id", "userId", "id"])
    new_email = _extract_from_body(body, ["new_email", "newEmail", "email"])
    if user_id is None or new_email is None:
        return False

    # Normalize and validate extracted values
    user_id_str = str(user_id)
    if not _is_valid_email(new_email):
        return False

    # Apply update to the simulated database
    return update_user_email(account_database, user_id_str, new_email)
