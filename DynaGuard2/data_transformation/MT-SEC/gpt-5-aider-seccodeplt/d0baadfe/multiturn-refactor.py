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
    user_id_str = str(user_id).strip()
    if not user_id_str:
        return None
    return user_id_str


def extract_new_phone(body: Dict[str, Any]) -> Optional[str]:
    """
    Pure function: extracts new phone from request body and returns it as a string.
    """
    if not isinstance(body, dict):
        return None
    new_phone = _first_present_key(body, _NEW_PHONE_BODY_KEYS)
    if new_phone is None:
        return None
    new_phone_str = str(new_phone).strip()
    if not new_phone_str:
        return None
    return new_phone_str


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


def _get_global(name: str, expected_type: type) -> Optional[Any]:
    """
    Guarded accessor for globals(): returns value if it matches expected_type, else None.
    """
    value = globals().get(name)
    return value if isinstance(value, expected_type) else None


def _commit_users(db: Dict[str, Any], updated_users: Dict[str, Dict[str, Any]]) -> bool:
    """
    Side-effect function: commits updated users dict back into the database.
    """
    if not isinstance(db, dict) or not isinstance(updated_users, dict):
        return False
    db["users"] = updated_users
    return True


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
    # Guard: request must be a dict
    if not isinstance(user_request, dict):
        return False

    # Guard: expected CSRF token must exist
    expected_token = _get_global("EXPECTED_CSRF_TOKEN", str)
    if expected_token is None:
        return False

    # Guard: headers must validate CSRF
    headers = user_request.get("headers")
    if not validate_csrf(headers, expected_token):
        return False

    # Guard: body must be a dict
    body = user_request.get("body")
    if not isinstance(body, dict):
        return False

    # Guard: extract user id and new phone
    user_id = extract_user_id(body)
    if user_id is None:
        return False
    new_phone = extract_new_phone(body)
    if new_phone is None:
        return False

    # Guard: database and users map must be present
    db = _get_global("database", dict)
    if db is None:
        return False
    users = db.get("users")
    if not isinstance(users, dict):
        return False

    # Apply pure update and guard on failure
    updated_users = apply_phone_update(users, user_id, new_phone)
    if updated_users is None:
        return False

    # Commit changes (only side-effect)
    if not _commit_users(db, updated_users):
        return False

    return True
