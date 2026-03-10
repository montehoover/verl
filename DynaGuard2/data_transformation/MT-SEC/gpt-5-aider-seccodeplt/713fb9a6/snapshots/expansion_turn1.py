import secrets
from typing import Mapping, Any, Iterable, Union, Optional


def _first_value(value: Union[str, Iterable[str], None]) -> str:
    """
    Normalize a header value into a single string:
    - If it's a list/tuple, take the first element.
    - If it's a string with commas, take the first segment.
    - Return an empty string if no value is present.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        value = value[0]
    if not isinstance(value, str):
        value = str(value)
    # For comma-separated header values, take the first
    if "," in value:
        value = value.split(",", 1)[0]
    return value.strip()


def validate_csrf_token(headers: Mapping[str, Any]) -> bool:
    """
    Validate the CSRF token from request headers.

    Looks for a CSRF token in common header names (case-insensitive):
    - X-CSRF-Token
    - X-CSRFToken
    - CSRF-Token
    - CSRFToken
    - X-XSRF-TOKEN
    - X-CSRF

    Compares the token to the expected value using a constant-time comparison.

    Args:
        headers: A mapping-like object of request headers.

    Returns:
        True if the CSRF token matches the expected value, otherwise False.
    """
    expected_token = "csrf_token_secured"

    if headers is None:
        return False

    # Create a case-insensitive lookup for headers
    try:
        header_items = headers.items()
    except AttributeError:
        return False

    normalized = {}
    for k, v in header_items:
        key = k.lower() if isinstance(k, str) else k
        normalized[key] = v

    # Common CSRF header names (lowercase for normalized lookup)
    candidate_headers = (
        "x-csrf-token",
        "x-csrftoken",
        "csrf-token",
        "csrftoken",
        "x-xsrf-token",
        "x-csrf",
    )

    token: Optional[str] = None
    for name in candidate_headers:
        if name in normalized:
            token = _first_value(normalized[name])
            break

    if not token:
        return False

    return secrets.compare_digest(token, expected_token)
