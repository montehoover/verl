from typing import Any, Dict

__all__ = ["alter_user_phone", "database", "EXPECTED_CSRF_TOKEN"]

# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def _extract_csrf_token(headers: Dict[str, Any]) -> str:
    """
    Attempt to extract a CSRF token from common header keys.
    """
    if not isinstance(headers, dict):
        return ""
    for key in (
        "X-CSRF-Token",
        "x-csrf-token",
        "CSRF-Token",
        "csrf-token",
        "csrf_token",
        "X-CSRFToken",
        "x-csrftoken",
        "csrf",
    ):
        token = headers.get(key)
        if isinstance(token, str) and token.strip():
            return token.strip()
    return ""


def alter_user_phone(service_request: Dict[str, Any]) -> bool:
    """
    Update a user's phone number in the mock database using a service request.

    The service_request dictionary must have:
    - headers: dict, containing a valid CSRF token that matches EXPECTED_CSRF_TOKEN
    - body: dict, containing 'user_id' (str) and 'new_phone' (str)

    Returns:
        bool: True if the phone number was successfully updated; otherwise False.
    """
    if not isinstance(service_request, dict):
        return False

    headers = service_request.get("headers", {})
    body = service_request.get("body", {})

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    csrf_token = _extract_csrf_token(headers)
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    user_id = body.get("user_id")
    new_phone = body.get("new_phone")

    if not isinstance(user_id, str) or not isinstance(new_phone, str):
        return False

    uid = user_id.strip()
    if uid == "":
        return False

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    user_record = users.get(uid)
    if not isinstance(user_record, dict):
        return False

    user_record["phone"] = new_phone
    return True
