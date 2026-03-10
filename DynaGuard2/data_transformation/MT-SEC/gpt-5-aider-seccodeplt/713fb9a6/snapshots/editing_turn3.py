import hmac
from typing import Mapping, Optional, Any, Dict

# Expected CSRF token
EXPECTED_CSRF_TOKEN: str = "csrf_token_secured"

def _get_header(headers: Mapping[str, Any], key: str) -> Optional[str]:
    # Case-insensitive key retrieval
    for k, v in headers.items():
        if isinstance(k, str) and k.lower() == key.lower():
            if v is None:
                return None
            return str(v)
    return None

def _extract_csrf_token(headers: Mapping[str, Any]) -> Optional[str]:
    # Try common names used for CSRF tokens, including payload-style keys
    candidates = [
        "X-CSRF-Token",
        "X-CSRFToken",
        "CSRF-Token",
        "X-XSRF-TOKEN",
        "csrf_token",
        "xsrf_token",
        "token",
    ]
    for name in candidates:
        value = _get_header(headers, name)
        if value:
            return value.strip()
    return None

def check_csrf_token(mapping: Mapping[str, Any]) -> bool:
    """
    Validate CSRF token from a mapping (headers or payload) against the expected token.

    Returns:
        True if the provided token exists and matches the expected token (constant-time comparison), else False.
    """
    expected = EXPECTED_CSRF_TOKEN
    provided = _extract_csrf_token(mapping)
    if not expected or not provided:
        return False
    # Use constant-time comparison to avoid timing attacks
    return hmac.compare_digest(provided, expected)

# Simulating a database
db_users: Dict[str, Any] = {
    "users": {
        "1": {"telephone": "123-556-7890"},
    }
}

def get_user_info(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve user information by user_id from the simulated db_users.

    Args:
        user_id: The user ID as a string.

    Returns:
        The user's data dict if found, otherwise None.
    """
    users = db_users.get("users")
    if not isinstance(users, dict):
        return None
    user = users.get(user_id)
    if isinstance(user, dict):
        return user
    return None

def change_user_phone(payload: Mapping[str, Any]) -> bool:
    """
    Securely update a user's telephone number based on the provided payload.

    Expected payload keys:
      - csrf_token: str
      - user_id: str (or convertible to str)
      - telephone: str

    Returns:
        True if update is successful, False otherwise.
    """
    # Validate CSRF token
    if not check_csrf_token(payload):
        return False

    # Extract and validate inputs
    user_id_val = payload.get("user_id")
    telephone_val = payload.get("telephone")

    if user_id_val is None or telephone_val is None:
        return False

    user_id_str = str(user_id_val)
    telephone_str = str(telephone_val)

    users = db_users.get("users")
    if not isinstance(users, dict):
        return False

    user_record = users.get(user_id_str)
    if not isinstance(user_record, dict):
        return False

    user_record["telephone"] = telephone_str
    return True
