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
    for key in keys:
        if key in body:
            return body[key]
    return None


def _get_csrf_token(headers: Dict[str, Any]) -> Any:
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
    if not isinstance(email, str):
        return False
    # Simple email validation
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email) is not None


def edit_user_email(payload: dict) -> bool:
    """
    Modify a user's email address within a web application.

    Args:
        payload (dict): The incoming request object containing headers and the body,
                        where the body includes the user ID and the new email.

    Returns:
        bool: True if the email address was updated successfully, otherwise False.
    """
    if not isinstance(payload, dict):
        return False

    headers = payload.get("headers") or {}
    if not isinstance(headers, dict):
        return False

    token = _get_csrf_token(headers)
    if token != VALID_TOKEN:
        return False

    body = payload.get("body")
    if not isinstance(body, dict):
        return False

    # Extract user id and new email with flexible key support
    user_id = _extract_from_body(body, ["user_id", "userId", "id"])
    new_email = _extract_from_body(body, ["new_email", "newEmail", "email"])

    if user_id is None or new_email is None:
        return False

    # Normalize user_id to string to match database keys
    user_id_str = str(user_id)

    if not _is_valid_email(new_email):
        return False

    try:
        users = account_database.get("users", {})
        if user_id_str not in users or not isinstance(users[user_id_str], dict):
            return False

        users[user_id_str]["email"] = new_email
        return True
    except Exception:
        return False
