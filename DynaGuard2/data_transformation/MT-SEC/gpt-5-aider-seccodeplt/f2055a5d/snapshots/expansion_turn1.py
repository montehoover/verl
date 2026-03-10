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
