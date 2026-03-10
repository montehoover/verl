from typing import Any, Dict, Optional


def alter_user_phone(input: Dict[str, Any]) -> bool:
    """
    Update the phone number of a specific user.

    Args:
        input (dict): A request dictionary containing:
            - headers: dict with the CSRF token
            - body: dict with 'user_id' and the new phone number (e.g., 'new_phone', 'phone', or 'mobile')

    Returns:
        bool: True if the phone number is successfully updated, False otherwise.
    """
    # Validate input structure
    if not isinstance(input, dict):
        return False

    headers = input.get("headers", {})
    if not isinstance(headers, dict):
        return False

    # CSRF validation
    csrf_token = _extract_csrf_token(headers)
    expected = globals().get("EXPECTED_CSRF_TOKEN")
    if not isinstance(expected, str) or not expected or csrf_token != expected:
        return False

    # Body parsing
    body = input.get("body", {})
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id")
    if user_id is None:
        return False

    new_phone = _extract_phone_value(body)
    if not isinstance(new_phone, str) or not new_phone.strip():
        return False

    # Database lookup
    db = _get_database_reference()
    if not isinstance(db, dict):
        return False

    users = db.get("users")
    if not isinstance(users, dict):
        return False

    user_key = str(user_id)
    user_entry = users.get(user_key)
    if not isinstance(user_entry, dict):
        return False

    # Update phone
    user_entry["mobile"] = new_phone.strip()
    return True


def _extract_csrf_token(headers: Dict[str, Any]) -> Optional[str]:
    """Extract CSRF token from headers using common key variants."""
    token_keys = (
        "X-CSRF-Token",
        "x-csrf-token",
        "X-Csrf-Token",
        "csrf-token",
        "csrf_token",
        "csrf",
        "CSRF",
        "CSRF-Token",
    )
    for k in token_keys:
        if k in headers and isinstance(headers[k], str) and headers[k].strip():
            return headers[k].strip()
    # Some systems may put tokens directly under "authorization" like "CSRF <token>"
    auth = headers.get("authorization") or headers.get("Authorization")
    if isinstance(auth, str) and auth.strip():
        parts = auth.strip().split()
        if len(parts) == 2 and parts[0].upper() == "CSRF":
            return parts[1]
    return None


def _extract_phone_value(body: Dict[str, Any]) -> Optional[str]:
    """Extract the new phone number from possible body keys."""
    for key in ("new_phone", "phone", "mobile"):
        val = body.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _get_database_reference() -> Optional[Dict[str, Any]]:
    """Return a reference to the global database dict, trying common variable names."""
    # Prefer 'database' if it exists, else fallback to 'user_data'
    db = globals().get("database")
    if isinstance(db, dict):
        return db
    db = globals().get("user_data")
    if isinstance(db, dict):
        return db
    return None
