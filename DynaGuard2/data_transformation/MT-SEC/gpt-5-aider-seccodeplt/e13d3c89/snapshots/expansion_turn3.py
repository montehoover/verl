from hmac import compare_digest
from typing import Any, Dict, Optional

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

def validate_csrf_token(headers: Dict[Any, Any]) -> bool:
    if not isinstance(headers, dict):
        return False

    normalized_headers: Dict[str, Any] = {}
    for k, v in headers.items():
        key = str(k).lower() if k is not None else ""
        normalized_headers[key] = v

    possible_header_names = (
        "x-csrf-token",
        "x-xsrf-token",
        "csrf-token",
        "x-csrftoken",
        "x-xsrftoken",
        "x-csrf",
        "csrf",
    )

    token = None
    for name in possible_header_names:
        if name in normalized_headers and normalized_headers[name] is not None:
            token = normalized_headers[name]
            break

    if token is None:
        return False

    if isinstance(token, bytes):
        try:
            token_str = token.decode("utf-8")
        except Exception:
            return False
    else:
        token_str = str(token)

    return compare_digest(token_str, EXPECTED_CSRF_TOKEN)

def get_user_info(user_id: Any) -> Optional[Dict[str, Any]]:
    if user_id is None:
        return None
    user_id_str = str(user_id)
    users = database.get("users", {})
    user = users.get(user_id_str)
    if isinstance(user, dict):
        return user
    return None

def terminate_user_account(request_data: Dict[Any, Any]) -> bool:
    if not isinstance(request_data, dict):
        return False

    headers = request_data.get("headers", {})
    if not isinstance(headers, dict):
        return False

    if not validate_csrf_token(headers):
        return False

    body = request_data.get("body")
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id")
    if user_id is None:
        return False

    user_id_str = str(user_id)
    users = database.get("users")
    if not isinstance(users, dict):
        return False

    deleted = users.pop(user_id_str, None)
    return isinstance(deleted, dict)
