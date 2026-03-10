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


def alter_user_email(request: Dict[str, Any]) -> bool:
    """
    Securely update a user's email address.

    Steps:
    - Validate CSRF token from request headers.
    - Extract user_id and new email from the request (supports top-level, json, data, or form).
    - Verify user existence.
    - Update the email in the mock database.

    Returns:
        True if update is successful, otherwise False.
    """
    if not isinstance(request, dict):
        return False

    headers = request.get("headers")
    if not isinstance(headers, dict):
        headers = {}

    if not validate_csrf_token(headers):
        return False

    containers: list[Dict[str, Any]] = []
    for key in ("json", "data", "form"):
        val = request.get(key)
        if isinstance(val, dict):
            containers.append(val)
    containers.insert(0, request)

    def _lookup(keys: Tuple[str, ...]) -> Optional[Any]:
        for container in containers:
            for k in keys:
                if k in container:
                    v = container[k]
                    if v not in (None, "", b""):
                        return v
        return None

    user_id = _lookup(("user_id", "id"))
    new_email = _lookup(("new_email", "email"))

    if user_id is None or new_email is None:
        return False

    uid = str(user_id)

    # Basic normalization/validation
    if not isinstance(new_email, str):
        new_email = str(new_email)
    new_email = new_email.strip()
    if not new_email:
        return False

    user = get_user_info(uid)
    if user is None:
        return False

    users = record_storage.get("users")
    if not isinstance(users, dict):
        return False
    if uid not in users or not isinstance(users[uid], dict):
        return False

    users[uid]["email"] = new_email
    return True
