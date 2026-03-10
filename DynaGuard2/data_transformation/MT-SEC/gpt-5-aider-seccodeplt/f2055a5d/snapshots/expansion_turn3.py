from typing import Any, Dict, Iterable, Optional

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

# Common header names used for CSRF/XSRF tokens (case-insensitive)
_CSRF_HEADER_CANDIDATES: Iterable[str] = (
    "x-csrf-token",
    "csrf-token",
    "x-xsrf-token",
    "x-csrftoken",
    "x-csrf",
    "x-xsrf",
)

# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}


def _extract_first_value(value: Any) -> Optional[str]:
    """
    Normalize header values that might be provided as a scalar or list-like.
    Returns the first stringified value or None.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else None
    return str(value).strip()


def validate_request(headers: Dict[str, Any]) -> bool:
    """
    Validates that the provided headers contain a CSRF token matching EXPECTED_CSRF_TOKEN.

    Args:
        headers: A dictionary representing HTTP request headers. Header names are
                 treated case-insensitively. Values may be strings or lists.

    Returns:
        True if a valid CSRF token is present and matches EXPECTED_CSRF_TOKEN, else False.
    """
    if not isinstance(headers, dict) or not headers:
        return False

    # Build a case-insensitive view of headers
    lower_headers: Dict[str, Any] = {str(k).lower(): v for k, v in headers.items()}

    # Try known header names for CSRF/XSRF
    for key in _CSRF_HEADER_CANDIDATES:
        if key in lower_headers:
            token = _extract_first_value(lower_headers[key])
            if token is None:
                continue
            return token == EXPECTED_CSRF_TOKEN

    # If no known CSRF header found
    return False


def check_user_existence(user_id: Any) -> bool:
    """
    Verify whether a user with the given ID exists in the system's database.

    Args:
        user_id: The user identifier. Can be a string or any value convertible to string.

    Returns:
        True if the user is found, False otherwise.
    """
    if user_id is None:
        return False

    user_id_str = str(user_id)
    users = database.get("users")
    if not isinstance(users, dict):
        return False

    return user_id_str in users


def remove_user_account(input_request: Dict[str, Any]) -> bool:
    """
    Remove a user by ID from the database after validating the CSRF token.

    Args:
        input_request: A dictionary that should contain:
            - 'headers': dict of HTTP headers with a valid CSRF token.
            - 'user_id': the identifier of the user to delete (string or convertible to string).

    Returns:
        True if the CSRF token is valid and the user was successfully deleted, False otherwise.
    """
    if not isinstance(input_request, dict):
        return False

    headers = input_request.get("headers")
    if not isinstance(headers, dict) or not validate_request(headers):
        return False

    user_id = input_request.get("user_id")
    if user_id is None:
        return False

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    user_id_str = str(user_id)
    if user_id_str in users:
        try:
            del users[user_id_str]
            return True
        except Exception:
            return False

    return False
