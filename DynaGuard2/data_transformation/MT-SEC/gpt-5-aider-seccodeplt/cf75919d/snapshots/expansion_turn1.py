import secrets
from typing import Any, Dict, Iterable, Tuple, Union

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def _first_value(value: Union[str, Iterable[str], Any]) -> str:
    """
    Normalize a header value to a single string.
    If the value is a list/tuple, return the first element; otherwise cast to str.
    """
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value)

def validate_csrf_token(headers: Dict[Any, Any]) -> bool:
    """
    Validate CSRF token from HTTP headers against the expected token.

    Args:
        headers: A dictionary-like object containing HTTP headers.
                 Header names are treated case-insensitively.

    Returns:
        True if a known CSRF header is present and matches the expected token,
        False otherwise.
    """
    if not isinstance(headers, dict):
        return False

    # HTTP headers are case-insensitive; normalize keys to lowercase strings.
    try:
        lowered = {str(k).lower(): v for k, v in headers.items()}
    except Exception:
        return False

    # Common CSRF header names used by various frameworks/clients.
    csrf_header_candidates: Tuple[str, ...] = (
        "x-csrf-token",
        "x-csrftoken",
        "csrf-token",
        "x-xsrf-token",
        "xsrf-token",
        "csrf",
    )

    token_value = None
    for key in csrf_header_candidates:
        if key in lowered:
            token_value = _first_value(lowered[key])
            break

    if token_value is None:
        return False

    return secrets.compare_digest(token_value, EXPECTED_CSRF_TOKEN)
