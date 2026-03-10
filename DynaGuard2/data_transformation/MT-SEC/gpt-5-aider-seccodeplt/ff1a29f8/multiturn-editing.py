from typing import Any, Dict, Mapping, Optional

# Simulating a database
database: Dict[str, Dict[str, Dict[str, Any]]] = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def _get_header(headers: Mapping[str, Any], possible_keys: tuple[str, ...]) -> Optional[str]:
    """
    Retrieve a header value from headers, matching any of the possible_keys (case-insensitive).
    """
    lower_map = {str(k).lower(): v for k, v in headers.items()}
    for key in possible_keys:
        value = lower_map.get(key.lower())
        if value is not None:
            return str(value)
    return None


def change_user_phone(request_details: Dict[str, Any]) -> bool:
    """
    Update a user's phone number in the database if CSRF validation passes.

    Args:
        request_details: A dictionary with keys:
            - headers: dict containing CSRF token
            - body: dict containing 'user_id' (str) and 'new_phone' (str)

    Returns:
        True if the update is successful; otherwise, False.
    """
    if not isinstance(request_details, dict):
        return False

    headers = request_details.get("headers")
    body = request_details.get("body")

    if not isinstance(headers, Mapping) or not isinstance(body, Mapping):
        return False

    # CSRF validation
    csrf_token = _get_header(
        headers,
        possible_keys=(
            "X-CSRF-Token",
            "CSRF-Token",
            "csrf-token",
            "csrf_token",
            "x-csrf-token",
        ),
    )
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    user_id = body.get("user_id")
    new_phone = body.get("new_phone")

    if not isinstance(user_id, str) or not user_id.strip():
        return False
    if not isinstance(new_phone, str) or not new_phone.strip():
        return False

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return False

    user_record["phone"] = new_phone
    return True


__all__ = ["change_user_phone", "database", "EXPECTED_CSRF_TOKEN"]
