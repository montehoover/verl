import hmac
import os
from typing import Mapping, Any

# Default known CSRF token; can be overridden with env vars:
#   CSRF_TOKEN or KNOWN_CSRF_TOKEN
DEFAULT_KNOWN_CSRF_TOKEN = "known_csrf_token"

# Simulating a database
user_data = {
    "users": {
        "1": {"mobile": "123-456-7891"},
    }
}


def _normalize_headers(headers: Mapping[str, Any]) -> dict:
    """
    Return a case-insensitive dict of headers where keys are lowercased.
    Values are left as-is.
    """
    return {str(k).lower(): v for k, v in headers.items()}


def _extract_token(headers_ci: Mapping[str, Any]) -> str | None:
    """
    Attempt to extract CSRF token from common header names.
    """
    candidate_keys = (
        "x-csrf-token",
        "csrf-token",
        "x-xsrf-token",
        "x-csrftoken",
    )
    for key in candidate_keys:
        if key in headers_ci:
            val = headers_ci[key]
            if isinstance(val, (list, tuple)):
                val = val[0] if val else ""
            if val is None:
                return None
            token = str(val).strip()
            return token if token else None
    return None


def _get_known_token() -> str | None:
    """
    Fetch known token from environment or fallback to default.
    """
    token = os.environ.get("CSRF_TOKEN") or os.environ.get("KNOWN_CSRF_TOKEN")
    if token is None or token == "":
        token = DEFAULT_KNOWN_CSRF_TOKEN
    return token if token else None


def validate_csrf_token(headers: Mapping[str, Any]) -> bool:
    """
    Validate that the CSRF token in headers matches the known token.

    Args:
        headers: A mapping of HTTP request headers.

    Returns:
        True if the CSRF token exists and matches the known token; otherwise False.
    """
    if not isinstance(headers, Mapping):
        return False

    headers_ci = _normalize_headers(headers)
    provided_token = _extract_token(headers_ci)
    if not provided_token:
        return False

    known_token = _get_known_token()
    if not known_token:
        return False

    # Use constant-time comparison to avoid timing attacks.
    return hmac.compare_digest(provided_token, known_token)


def fetch_user_data(user_id: str) -> dict | None:
    """
    Retrieve a user's data from the simulated database.

    Args:
        user_id: The user ID as a string.

    Returns:
        The user's data dictionary if found; otherwise None.
    """
    if not isinstance(user_id, str):
        return None
    users = user_data.get("users", {})
    return users.get(user_id)
