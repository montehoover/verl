from typing import Optional, Dict, Any

# Header and body keys to search
_CSRF_HEADER_KEYS = ("X-CSRF-Token", "x-csrf-token", "CSRF-Token", "csrf_token", "csrf")
_USER_ID_BODY_KEYS = ("user_id", "id", "userId")
_NEW_PHONE_BODY_KEYS = ("new_phone", "phone", "newPhone", "new_phone_number", "phone_number")


def _first_present_key(d: Dict[str, Any], keys: tuple) -> Optional[Any]:
    """
    Pure helper to get the first present value for any of the keys from the dict.
    """
    if not isinstance(d, dict):
        return None
    for key in keys:
        if key in d:
            return d.get(key)
    return None


def validate_csrf(headers: Dict[str, Any], expected_token: str) -> bool:
    """
    Pure function: validates CSRF token from headers against the expected token.
    """
    if not isinstance(headers, dict) or not isinstance(expected_token, str):
        return False
    csrf_token = _first_present_key(headers, _CSRF_HEADER_KEYS)
    if not isinstance(csrf_token, str):
        return False
    return csrf_token == expected_token


def extract_user_id(body: Dict[str, Any]) -> Optional[str]:
    """
    Pure function: extracts user id from request body and returns it as a string.
    """
    if not isinstance(body, dict):
        return None
    user_id = _first_present_key(body, _USER_ID_BODY_KEYS)
    if user_id is None:
        return None
    return str(user_id)


def extract_new_phone(body: Dict[str, Any]) -> Optional[str]:
    """
    Pure function: extracts new phone from request body and returns it as a string.
    """
    if not isinstance(body, dict):
        return None
    new_phone = _first_present_key(body, _NEW_PHONE_BODY_KEYS)
    if new_phone is None:
        return None
    return str(new_phone)


def apply_phone_update(users: Dict[str, Dict[str, Any]], user_id: str, new_phone: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Pure function: returns a new users dict with the user's phone updated if possible.
    Does not mutate the input users dict. Returns None if update cannot be applied.
    """
    if not isinstance(users, dict) or not isinstance(user_id, str) or not isinstance(new_phone, str):
        return None

    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return None

    # Create shallow copies to preserve purity
    new_users = dict(users)
    new_record = dict(user_record)
    new_record["phone"] = new_phone
    new_users[user_id] = new_record
    return new_users


def modify_user_phone(user_request: dict) -> bool:
    """
    Modify a user's phone number in the global 'database' using data from user_request.

    Expects:
      - Global variables:
          database: dict
          EXPECTED_CSRF_TOKEN: str
      - user_request: dict with keys:
          headers: dict containing a CSRF token
          body: dict containing user ID and new phone number

    Returns:
      - True if update succeeded, otherwise False.
    """
    # Validate input type
    if not isinstance(user_request, dict):
        return False

    # Validate and check CSRF
    headers = user_request.get("headers")
    expected_token = globals().get("EXPECTED_CSRF_TOKEN")
    if not isinstance(expected_token, str):
        return False
    if not validate_csrf(headers, expected_token):
        return False

    # Validate body
    body = user_request.get("body")
    if not isinstance(body, dict):
        return False

    # Extract user ID and new phone
    user_id = extract_user_id(body)
    new_phone = extract_new_phone(body)
    if user_id is None or new_phone is None:
        return False

    # Validate database structure and apply update using pure helper
    db = globals().get("database")
    if not isinstance(db, dict):
        return False
    users = db.get("users")
    if not isinstance(users, dict):
        return False

    updated_users = apply_phone_update(users, user_id, new_phone)
    if updated_users is None:
        return False

    # Commit the pure result to the global database (side-effect only here)
    db["users"] = updated_users
    return True
