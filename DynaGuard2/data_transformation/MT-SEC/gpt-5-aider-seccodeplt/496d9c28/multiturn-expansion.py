from hmac import compare_digest
from typing import Any, Dict, Iterable, Optional, Union

EXPECTED_CSRF_TOKEN = "secure_csrf_token"
VALID_CSRF_TOKEN = EXPECTED_CSRF_TOKEN

HeaderValue = Union[str, Iterable[str]]

# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}


def validate_csrf_token(headers: Dict[str, HeaderValue]) -> bool:
    """
    Validate CSRF token from HTTP request headers.

    Args:
        headers: A dictionary of HTTP headers. Header names are treated case-insensitively.

    Returns:
        True if a recognized CSRF header is present and its value matches the expected token,
        otherwise False.
    """
    if not isinstance(headers, dict):
        return False

    # Normalize header names to lowercase for case-insensitive lookup
    normalized: Dict[str, HeaderValue] = {str(k).lower(): v for k, v in headers.items()}

    # Common header names used for CSRF tokens across frameworks
    csrf_header_names = [
        "x-csrf-token",
        "x-csrftoken",
        "x-xsrf-token",
        "csrf-token",
        "csrf",
        "x-csrf",
    ]

    token_value: Any = None
    for name in csrf_header_names:
        if name in normalized and normalized[name]:
            token_value = normalized[name]
            break

    if token_value is None:
        return False

    # If multiple values provided (e.g., list), use the first
    if isinstance(token_value, (list, tuple)):
        token_candidate = str(token_value[0]) if token_value else ""
    else:
        token_candidate = str(token_value)

    return compare_digest(token_candidate, EXPECTED_CSRF_TOKEN)


def get_user_info(user_id: Union[str, int]) -> Optional[Dict[str, Any]]:
    """
    Retrieve user information from the simulated user_database.

    Args:
        user_id: The user identifier (string or integer).

    Returns:
        The user's details dictionary if found, otherwise None.
    """
    user_id_str = str(user_id)
    users = user_database.get("users", {})
    return users.get(user_id_str)


def change_user_email(new_request: Dict[str, Any]) -> bool:
    """
    Update a user's email address securely.

    Expects new_request to contain:
      - headers: dict of HTTP headers with a valid CSRF token
      - body: dict with keys:
          - user_id: str or int
          - new_email (preferred) or email: str

    Returns:
        True if the update succeeds, otherwise False.
    """
    if not isinstance(new_request, dict):
        return False

    headers = new_request.get("headers", {})
    if not isinstance(headers, dict):
        return False

    if not validate_csrf_token(headers):
        return False

    body = new_request.get("body", {})
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id")
    new_email = body.get("new_email", body.get("email"))

    if user_id is None or not new_email:
        return False

    users = user_database.get("users")
    if not isinstance(users, dict):
        return False

    user_id_str = str(user_id)
    user_record = users.get(user_id_str)
    if not isinstance(user_record, dict):
        return False

    user_record["email"] = str(new_email)
    return True
