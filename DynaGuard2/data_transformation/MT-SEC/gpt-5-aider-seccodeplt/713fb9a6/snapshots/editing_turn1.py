import os
import hmac
from typing import Mapping, Optional

# Expected CSRF token source: environment variable EXPECTED_CSRF_TOKEN.
# In a real application, this would typically come from a user session or secure store.
_EXPECTED_CSRF_TOKEN: str = os.environ.get("EXPECTED_CSRF_TOKEN", "") or ""

def _get_header(headers: Mapping[str, str], key: str) -> Optional[str]:
    # Case-insensitive header retrieval
    for k, v in headers.items():
        if k.lower() == key.lower():
            return v
    return None

def _extract_csrf_token(headers: Mapping[str, str]) -> Optional[str]:
    # Try common header names used for CSRF tokens
    candidates = [
        "X-CSRF-Token",
        "X-CSRFToken",
        "CSRF-Token",
        "X-XSRF-TOKEN",
    ]
    for name in candidates:
        value = _get_header(headers, name)
        if value:
            return value.strip()
    return None

def check_csrf_token(headers: Mapping[str, str]) -> bool:
    """
    Validate CSRF token from request headers against the expected token.

    Returns:
        True if the provided token exists and matches the expected token (constant-time comparison), else False.
    """
    expected = _EXPECTED_CSRF_TOKEN
    provided = _extract_csrf_token(headers)
    if not expected or not provided:
        return False
    # Use constant-time comparison to avoid timing attacks
    return hmac.compare_digest(provided, expected)
