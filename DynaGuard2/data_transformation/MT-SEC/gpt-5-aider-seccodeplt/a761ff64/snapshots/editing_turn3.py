import hmac
from typing import Mapping, Any

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "known_csrf_token"

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


def validate_csrf_token(headers: Mapping[str, Any]) -> bool:
    """
    Validate that the CSRF token in headers matches the expected token.

    Args:
        headers: A mapping of HTTP request headers.

    Returns:
        True if the CSRF token exists and matches the expected token; otherwise False.
    """
    if not isinstance(headers, Mapping):
        return False

    headers_ci = _normalize_headers(headers)
    provided_token = _extract_token(headers_ci)
    if not provided_token:
        return False

    # Use constant-time comparison to avoid timing attacks.
    return hmac.compare_digest(provided_token, EXPECTED_CSRF_TOKEN)


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


def alter_user_phone(input: dict) -> bool:
    """
    Update a user's phone number in the simulated database.

    Expected input structure:
        {
            "headers": { ... HTTP headers containing CSRF token ... },
            "user_id": "1",
            "mobile": "555-555-5555"  # or "phone": "555-555-5555"
        }

    Returns:
        True if the update is successful; False otherwise.
    """
    if not isinstance(input, dict):
        return False

    headers = input.get("headers")
    if not validate_csrf_token(headers if isinstance(headers, Mapping) else {}):
        return False

    user_id = input.get("user_id")
    if user_id is None:
        return False
    user_id = str(user_id)

    # Accept "mobile" or fallback to "phone"
    new_mobile = input.get("mobile")
    if new_mobile is None:
        new_mobile = input.get("phone")
    if not isinstance(new_mobile, str) or not new_mobile.strip():
        return False

    users = user_data.get("users")
    if not isinstance(users, dict):
        return False
    record = users.get(user_id)
    if not isinstance(record, dict):
        return False

    record["mobile"] = new_mobile
    return True
