import hmac
from typing import Any, Dict, Iterable, Optional


EXPECTED_CSRF_TOKEN = "secure_csrf_token"
CSRF_HEADER_CANDIDATES: Iterable[str] = (
    "x-csrf-token",
    "x-csrftoken",
    "csrf-token",
    "x-xsrf-token",
)

# Simulating a database
db = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}


def validate_csrf_token(headers: Dict[str, Any]) -> bool:
    """
    Validate the CSRF token from request headers.

    - Accepts a dictionary of headers (case-insensitive on keys).
    - Looks for common CSRF header names.
    - Compares using constant-time comparison.
    """
    if not isinstance(headers, dict):
        return False

    token_value = None
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() in CSRF_HEADER_CANDIDATES:
            token_value = value
            break

    if token_value is None:
        return False

    # If multiple values provided, use the first one
    if isinstance(token_value, (list, tuple)):
        if not token_value:
            return False
        token_value = token_value[0]

    if token_value is None:
        return False

    token_str = str(token_value).strip()
    return hmac.compare_digest(token_str, EXPECTED_CSRF_TOKEN)


def get_user_info(user_id: Any) -> Optional[Dict[str, Any]]:
    """
    Retrieve user details from the simulated database.

    - Accepts any type for user_id; it will be coerced to string for lookup.
    - Returns a dictionary of user details if found, otherwise None.
    """
    if user_id is None:
        return None

    user_id_str = str(user_id)
    users = db.get("users")
    if not isinstance(users, dict):
        return None

    return users.get(user_id_str)


def modify_user_phone(client_request: Dict[str, Any]) -> bool:
    """
    Securely update a user's phone number.

    Expects client_request to be a dict with:
      - headers: dict containing CSRF token in one of the accepted header names
      - body: dict containing user identifier and new phone number:
          possible id keys: 'user_id', 'id', 'userId'
          possible phone keys: 'new_phone', 'phone', 'newPhone'
    Returns True if update succeeds; otherwise False.
    """
    if not isinstance(client_request, dict):
        return False

    headers = client_request.get("headers")
    if not isinstance(headers, dict):
        return False

    if not validate_csrf_token(headers):
        return False

    body = client_request.get("body")
    if not isinstance(body, dict):
        return False

    def _extract_value(container: Dict[str, Any], keys: Iterable[str]) -> Any:
        for k in keys:
            if k in container:
                return container[k]
        return None

    raw_user_id = _extract_value(body, ("user_id", "id", "userId"))
    raw_new_phone = _extract_value(body, ("new_phone", "phone", "newPhone"))

    # Normalize potential list/tuple values
    if isinstance(raw_user_id, (list, tuple)):
        raw_user_id = raw_user_id[0] if raw_user_id else None
    if isinstance(raw_new_phone, (list, tuple)):
        raw_new_phone = raw_new_phone[0] if raw_new_phone else None

    if raw_user_id is None or raw_new_phone is None:
        return False

    user_id_str = str(raw_user_id).strip()
    new_phone_str = str(raw_new_phone).strip()

    if not user_id_str or not new_phone_str:
        return False

    user_record = get_user_info(user_id_str)
    if not isinstance(user_record, dict):
        return False

    # Update the user's phone
    user_record["phone"] = new_phone_str
    return True
