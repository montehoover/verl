import hmac
from typing import Any, Dict, Iterable, Optional

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

_CSRF_HEADER_CANDIDATES: Iterable[str] = (
    "x-csrf-token",
    "x-csrf",
    "csrf-token",
    "csrf",
    "x-xsrf-token",
    "xsrf-token",
)

# Simulating a database
database: Dict[str, Any] = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}


def _extract_header_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        return str(value)
    except Exception:
        return None


def validate_request(headers: Dict[str, Any]) -> bool:
    if not isinstance(headers, dict):
        return False

    lower_headers: Dict[str, Any] = {}
    for k, v in headers.items():
        if isinstance(k, str):
            lower_headers[k.lower()] = v

    token: Optional[str] = None
    for key in _CSRF_HEADER_CANDIDATES:
        if key in lower_headers:
            token = _extract_header_value(lower_headers[key])
            break

    if token is None:
        return False

    return hmac.compare_digest(token, EXPECTED_CSRF_TOKEN)


def get_user_info(user_id: Any) -> Optional[Dict[str, Any]]:
    if user_id is None:
        return None
    try:
        user_key = str(user_id)
    except Exception:
        return None

    users = database.get("users")
    if not isinstance(users, dict):
        return None

    user = users.get(user_key)
    if isinstance(user, dict):
        return user
    return None


def change_user_contact(transaction_request: Dict[str, Any]) -> bool:
    if not isinstance(transaction_request, dict):
        return False

    headers = transaction_request.get("headers")
    if not isinstance(headers, dict):
        return False

    if not validate_request(headers):
        return False

    body = transaction_request.get("body")
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id")
    new_phone = body.get("phone")

    if user_id is None or new_phone is None:
        return False

    user = get_user_info(user_id)
    if user is None or not isinstance(user, dict):
        return False

    try:
        user["phone"] = str(new_phone)
    except Exception:
        return False

    return True
