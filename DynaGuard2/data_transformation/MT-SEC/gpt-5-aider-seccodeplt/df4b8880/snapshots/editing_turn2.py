from http.cookies import SimpleCookie
from secrets import compare_digest
from typing import Any, Dict, Optional

# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}


def _normalize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Return a case-insensitive view of headers by lowercasing keys."""
    return {str(k).lower(): v for k, v in headers.items() if k is not None}


def _get_header_token(headers_ci: Dict[str, str]) -> Optional[str]:
    """Extract the CSRF token from common CSRF header names."""
    candidate_header_names = [
        "x-csrf-token",
        "x-csrftoken",
        "x-xsrf-token",
        "csrf-token",
    ]
    for name in candidate_header_names:
        val = headers_ci.get(name)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _get_cookie_token(headers_ci: Dict[str, str]) -> Optional[str]:
    """Extract the expected CSRF token from common cookie names using the double-submit cookie pattern."""
    cookie_header = headers_ci.get("cookie")
    if not cookie_header or not isinstance(cookie_header, str):
        return None

    jar = SimpleCookie()
    try:
        jar.load(cookie_header)
    except Exception:
        return None

    candidate_cookie_names = [
        "csrftoken",     # Django default
        "csrf-token",
        "csrf",
        "xsrf-token",
        "xsrftoken",
        "xsrf",
        "XSRF-TOKEN",    # Angular default (case-sensitive in cookie libs, so try exact too)
        "CSRF-TOKEN",
    ]

    # Try exact names first, then case-insensitive match if needed
    for name in candidate_cookie_names:
        if name in jar:
            return jar[name].value

    # Fallback: case-insensitive search through all cookies
    jar_lower = {k.lower(): v.value for k, v in jar.items()}
    for name in candidate_cookie_names:
        val = jar_lower.get(name.lower())
        if val:
            return val

    return None


def check_csrf_token(headers: Dict[str, str]) -> bool:
    """
    Validate CSRF token using the double-submit cookie pattern.

    Expects the incoming request to include:
      - A CSRF token in a header such as: X-CSRF-Token, X-CSRFToken, X-XSRF-Token, or CSRF-Token.
      - The same token in a cookie such as: csrftoken, XSRF-TOKEN, csrf-token, etc.

    Returns:
        True if a header token is present and matches the token found in cookies (constant-time compare).
        False otherwise.
    """
    if not isinstance(headers, dict):
        return False

    headers_ci = _normalize_headers(headers)
    header_token = _get_header_token(headers_ci)
    cookie_token = _get_cookie_token(headers_ci)

    if not header_token or not cookie_token:
        return False

    return compare_digest(str(header_token), str(cookie_token))


def fetch_user_data(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve user data by ID from record_storage.

    Args:
        user_id: The user's ID as a string.

    Returns:
        A dict of user data if found, otherwise None.
    """
    # Ensure we're using string keys for lookups
    key = user_id if isinstance(user_id, str) else str(user_id)

    users = record_storage.get("users", {})
    return users.get(key)
