from typing import Mapping, Any, Iterable, Optional, Union
import hmac

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}


def _lookup_header(headers: Mapping[str, Any], key: str) -> Any:
    key_lower = key.lower()
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == key_lower:
            return v
    return None


def _normalize_header_value(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    return stripped
    return None


def _extract_token_from_headers(headers: Mapping[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        raw = _lookup_header(headers, key)
        token = _normalize_header_value(raw)
        if token is not None:
            return token
    return None


def validate_csrf_token(headers: Mapping[str, Any]) -> bool:
    """
    Validate the CSRF token from request headers.

    Looks for a token in common CSRF header names and compares it to the expected token
    using a constant-time comparison.

    Returns True if valid, False otherwise.
    """
    if not isinstance(headers, Mapping):
        return False

    candidate_header_names = (
        "x-csrf-token",
        "x-csrftoken",
        "csrf-token",
        "x-xsrf-token",
    )

    token = _extract_token_from_headers(headers, candidate_header_names)
    if token is None or not isinstance(token, str):
        return False

    return hmac.compare_digest(token, EXPECTED_CSRF_TOKEN)


def get_user_info(user_id: Union[str, int]) -> Optional[Mapping[str, Any]]:
    """
    Retrieve user information from the simulated database.

    Returns the user details dict if found, otherwise None.
    """
    if user_id is None:
        return None

    users = database.get("users")
    if not isinstance(users, dict):
        return None

    user_key = str(user_id)
    user = users.get(user_key)
    if isinstance(user, dict):
        return user
    return None


def _extract_user_id(request: Mapping[str, Any]) -> Optional[Union[str, int]]:
    """
    Try to extract a user ID from a request-like mapping.
    Checks several common locations/keys.
    """
    candidate_keys = ("user_id", "userid", "userId", "id")
    # Top-level
    for key in candidate_keys:
        if key in request:
            val = request.get(key)
            if isinstance(val, (str, int)):
                return val
    # Nested common containers
    nested_containers = ("params", "data", "json", "body", "form", "path_params", "query", "query_params")
    for container_key in nested_containers:
        container = request.get(container_key)
        if isinstance(container, Mapping):
            for key in candidate_keys:
                if key in container:
                    val = container.get(key)
                    if isinstance(val, (str, int)):
                        return val
    return None


def remove_user_profile(request: Mapping[str, Any]) -> bool:
    """
    Remove a user's account from the database.

    Validates CSRF token from request["headers"], verifies the user exists,
    and deletes the record. Returns True if successfully removed, else False.
    """
    if not isinstance(request, Mapping):
        return False

    headers = request.get("headers")
    if not isinstance(headers, Mapping):
        headers = {}

    if not validate_csrf_token(headers):
        return False

    user_id = _extract_user_id(request)
    if user_id is None:
        return False

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    user_key = str(user_id)
    if user_key not in users:
        return False

    try:
        del users[user_key]
        return True
    except Exception:
        return False


__all__ = ["validate_csrf_token", "get_user_info", "remove_user_profile"]
