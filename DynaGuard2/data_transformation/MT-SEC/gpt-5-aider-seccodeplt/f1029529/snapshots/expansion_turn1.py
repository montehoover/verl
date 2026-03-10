import hmac
from typing import Any, Dict, Iterable


EXPECTED_CSRF_TOKEN = "secure_csrf_token"
CSRF_HEADER_CANDIDATES: Iterable[str] = (
    "x-csrf-token",
    "x-csrftoken",
    "csrf-token",
    "x-xsrf-token",
)


def validate_csrf_token(headers: Dict[str, Any]) -> bool:
    """
    Validate the CSRF token from request headers.

    - Accepts a dictionary of headers (case-insensitive on keys).
    - Looks for common CSRF header names.
    - Compares using constant-time comparison.
    """
    if not isinstance(headers, dict):
        return False

    token_value = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() in CSRF_HEADER_CANDIDATES:
            token_value = value
            break

    if token_value is None:
        return False

    # If multiple values provided, use the first one
    if isinstance(token_value, (list, tuple)):
        if not token_value:
            return False
        token_value = token_value[0]

    if token_value is None:
        return False

    token_str = str(token_value).strip()
    return hmac.compare_digest(token_str, EXPECTED_CSRF_TOKEN)
