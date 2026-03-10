from hmac import compare_digest
from typing import Any, Dict, Optional, Tuple, Union

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

# Common CSRF header names used by various frameworks/clients (case-insensitive).
_CSRF_HEADER_CANDIDATES: Tuple[str, ...] = (
    "x-csrf-token",
    "x-xsrf-token",
    "csrf-token",
    "x-csrftoken",
)

# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}


def _normalize_headers(headers: Dict[str, Any]) -> Dict[str, str]:
    """
    Normalize header keys to lowercase and coerce values to strings.

    - If a header value is a list/tuple, the first non-empty item is used (else empty string).
    - If a header value is bytes, it is decoded as UTF-8 (errors ignored).
    - None becomes an empty string.
    """
    normalized: Dict[str, str] = {}
    for key, value in headers.items():
        val: Optional[Union[str, bytes]] = None

        if isinstance(value, (list, tuple)):
            # Choose the first non-empty value if available
            for item in value:
                if item not in (None, "", b""):
                    val = item  # type: ignore[assignment]
                    break
            if val is None and value:
                val = value[0]  # type: ignore[assignment]
        else:
            val = value  # type: ignore[assignment]

        if isinstance(val, bytes):
            s = val.decode("utf-8", errors="ignore")
        elif val is None:
            s = ""
        else:
            s = str(val)

        normalized[str(key).lower()] = s
    return normalized


def validate_csrf_token(headers: Dict[str, Any]) -> bool:
    """
    Validate the CSRF token from request headers.

    Looks for a token in common CSRF header names and compares it (in constant-time)
    against the expected token.

    Args:
        headers: A dictionary representing the request headers.

    Returns:
        True if a matching CSRF token is present, otherwise False.
    """
    if not isinstance(headers, dict):
        return False

    normalized = _normalize_headers(headers)

    token: Optional[str] = None
    for candidate in _CSRF_HEADER_CANDIDATES:
        if candidate in normalized:
            candidate_value = normalized[candidate].strip()
            if candidate_value:
                token = candidate_value
                break

    if not token:
        return False

    return compare_digest(token, EXPECTED_CSRF_TOKEN)


def get_user_info(user_id: Union[str, int]) -> Optional[Dict[str, Any]]:
    """
    Retrieve user information from the mock database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        The user's data dictionary if found; otherwise, None.
    """
    if user_id is None:
        return None
    uid = str(user_id)
    users = record_storage.get("users") or {}
    return users.get(uid)
