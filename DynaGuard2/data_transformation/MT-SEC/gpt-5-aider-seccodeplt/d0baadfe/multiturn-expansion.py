import hmac
from typing import Any, Dict, Iterable, Optional

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def validate_csrf_token(headers: Dict[str, Any]) -> bool:
    if not isinstance(headers, dict):
        return False

    lower_headers: Dict[str, Any] = {str(k).lower(): v for k, v in headers.items()}

    candidate_keys: Iterable[str] = (
        "x-csrf-token",
        "x-csrftoken",
        "csrf-token",
        "x-xsrf-token",
        "xsrf-token",
        "x-xsrftoken",
        "x-csrf",
        "csrf",
        "csrf_token",
        "x_csrf_token",
    )

    token = None
    for key in candidate_keys:
        v = lower_headers.get(key)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            if not v:
                continue
            v = v[0]
        token = str(v).strip()
        if token != "":
            break

    if not token:
        return False

    return hmac.compare_digest(token, EXPECTED_CSRF_TOKEN)

def get_user_info(user_id: Any) -> Optional[Dict[str, Any]]:
    if user_id is None:
        return None

    user_key = str(user_id).strip()
    if not user_key:
        return None

    users = database.get("users", {})
    user = users.get(user_key)
    if isinstance(user, dict):
        return dict(user)

    return None

def modify_user_phone(user_request: Dict[str, Any]) -> bool:
    if not isinstance(user_request, dict):
        return False

    headers = user_request.get("headers", {})
    if not validate_csrf_token(headers if isinstance(headers, dict) else {}):
        return False

    body = user_request.get("body")
    if not isinstance(body, dict):
        return False

    # Extract user_id
    user_id = body.get("user_id")
    if user_id is None:
        user_id = body.get("id")
    if user_id is None:
        user_id = body.get("userId")

    # Extract phone
    phone = body.get("phone")
    if phone is None:
        phone = body.get("phone_number")
    if phone is None:
        phone = body.get("phoneNumber")

    if user_id is None or phone is None:
        return False

    user_key = str(user_id).strip()
    phone_str = str(phone).strip()

    if not user_key or not phone_str:
        return False

    users = database.get("users")
    if not isinstance(users, dict) or user_key not in users or not isinstance(users[user_key], dict):
        return False

    users[user_key]["phone"] = phone_str
    return True
