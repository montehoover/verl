import hmac
from typing import Any, Dict, Optional

# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token_value"

def validate_csrf_protection(headers: Dict[str, Any]) -> bool:
    if not isinstance(headers, dict):
        return False

    lowered: Dict[str, Any] = {}
    for k, v in headers.items():
        if isinstance(k, str):
            lowered[k.lower()] = v

    header_candidates = [
        "x-csrf-token",
        "x-csrftoken",
        "csrf-token",
        "x-xsrf-token",
        "x-xsrftoken",
    ]

    token = None
    for key in header_candidates:
        if key in lowered:
            token = lowered[key]
            break

    if token is None:
        return False

    if isinstance(token, (list, tuple)):
        token = token[0] if token else ""

    if not isinstance(token, str):
        token = str(token)

    token = token.strip()
    if not token:
        return False

    expected = EXPECTED_CSRF_TOKEN
    return hmac.compare_digest(token, expected)

def find_user_by_id(user_id: Any) -> Optional[Dict[str, Any]]:
    if user_id is None:
        return None

    try:
        user_key = str(user_id)
    except Exception:
        return None

    users = db_store.get("users") if isinstance(db_store, dict) else None
    if not isinstance(users, dict):
        return None

    user = users.get(user_key)
    if isinstance(user, dict):
        return user
    return None

def modify_user_phone(request_data: Dict[str, Any]) -> bool:
    # Validate request_data structure
    if not isinstance(request_data, dict):
        return False

    # CSRF validation
    headers = request_data.get("headers")
    if not isinstance(headers, dict):
        headers = {}
    if not validate_csrf_protection(headers):
        return False

    # Extract user_id
    user_id = request_data.get("user_id")
    if user_id is None:
        return False

    # Ensure user exists
    user = find_user_by_id(user_id)
    if not isinstance(user, dict):
        return False

    # Extract new phone value from potential keys
    new_phone = None
    for key in ("cell", "phone", "phone_number", "new_cell", "new_phone"):
        if key in request_data:
            new_phone = request_data.get(key)
            break

    if new_phone is None:
        return False

    if isinstance(new_phone, (list, tuple)):
        new_phone = new_phone[0] if new_phone else ""
    if not isinstance(new_phone, str):
        new_phone = str(new_phone)
    new_phone = new_phone.strip()

    if not new_phone:
        return False

    # Update the database
    users = db_store.get("users")
    if not isinstance(users, dict):
        return False

    user_key = str(user_id)
    if user_key not in users:
        return False

    users[user_key]["cell"] = new_phone
    return True
