import hmac
from typing import Mapping, Any, Iterable, Optional, Dict

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Common header names used for CSRF tokens across frameworks
_CSRF_HEADER_CANDIDATES: Iterable[str] = (
    "x-csrf-token",
    "x-csrftoken",
    "x-xsrf-token",
    "x-xsrftoken",
    "csrf-token",
    "csrftoken",
    "xsrf-token",
    "xsrftoken",
)


def validate_csrf_token(headers: Mapping[str, Any]) -> bool:
    """
    Validate CSRF token from request headers.

    Args:
        headers: A mapping/dict of request headers.

    Returns:
        True if the CSRF token matches the expected value, False otherwise.
    """
    if not isinstance(headers, Mapping):
        return False

    # Normalize header keys to lower-case for case-insensitive lookup
    lowered = {str(k).lower(): v for k, v in headers.items() if isinstance(k, str)}

    token = None
    for key in _CSRF_HEADER_CANDIDATES:
        if key in lowered:
            token = lowered[key]
            break

    if token is None:
        return False

    # Support cases where header value may be a list or tuple
    if isinstance(token, (list, tuple)):
        token = token[0] if token else ""

    # Ensure string type for comparison
    token_str = str(token)

    # Constant-time comparison to mitigate timing attacks
    return hmac.compare_digest(token_str, EXPECTED_CSRF_TOKEN)


def get_user_info(user_id: Any) -> Optional[Dict[str, Any]]:
    """
    Retrieve user information from the simulated database.

    Args:
        user_id: The ID of the user to retrieve. Can be any type; will be cast to string.

    Returns:
        The user's details dict if found, otherwise None.
    """
    if user_id is None:
        return None

    user_key = str(user_id)
    users = database.get("users") or {}
    return users.get(user_key)
