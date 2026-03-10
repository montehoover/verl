import secrets
from typing import Mapping, Any, Iterable, Union, Optional

# Simulating a database
db_users = {
    "users": {
        "1": {"telephone": "123-556-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "csrf_token_secured"


def _first_value(value: Union[str, Iterable[str], None]) -> str:
    """
    Normalize a header value into a single string:
    - If it's a list/tuple, take the first element.
    - If it's a string with commas, take the first segment.
    - Return an empty string if no value is present.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        value = value[0]
    if not isinstance(value, str):
        value = str(value)
    # For comma-separated header values, take the first
    if "," in value:
        value = value.split(",", 1)[0]
    return value.strip()


def validate_csrf_token(headers: Mapping[str, Any]) -> bool:
    """
    Validate the CSRF token from request headers.

    Looks for a CSRF token in common header names (case-insensitive):
    - X-CSRF-Token
    - X-CSRFToken
    - CSRF-Token
    - CSRFToken
    - X-XSRF-TOKEN
    - X-CSRF

    Compares the token to the expected value using a constant-time comparison.

    Args:
        headers: A mapping-like object of request headers.

    Returns:
        True if the CSRF token matches the expected value, otherwise False.
    """
    if headers is None:
        return False

    # Create a case-insensitive lookup for headers
    try:
        header_items = headers.items()
    except AttributeError:
        return False

    normalized = {}
    for k, v in header_items:
        key = k.lower() if isinstance(k, str) else k
        normalized[key] = v

    # Common CSRF header names (lowercase for normalized lookup)
    candidate_headers = (
        "x-csrf-token",
        "x-csrftoken",
        "csrf-token",
        "csrftoken",
        "x-xsrf-token",
        "x-csrf",
    )

    token: Optional[str] = None
    for name in candidate_headers:
        if name in normalized:
            token = _first_value(normalized[name])
            break

    if not token:
        return False

    return secrets.compare_digest(token, EXPECTED_CSRF_TOKEN)


def get_user_info(user_id: Union[str, int]) -> Optional[Mapping[str, Any]]:
    """
    Retrieve user details from the simulated database by user ID.

    Args:
        user_id: The user's identifier (string or integer).

    Returns:
        A mapping of the user's details if found; otherwise, None.
    """
    users = db_users.get("users", {})
    if not isinstance(users, dict):
        return None
    return users.get(str(user_id))


def change_user_phone(request: Mapping[str, Any]) -> bool:
    """
    Update a user's phone number securely.

    The request is expected to contain:
    - headers: Mapping[str, Any] with a CSRF token header
    - user_id: str|int (may also be nested in request['user']['id'] or request['params']['user_id'])
    - telephone or new_phone: the new phone number (may also be nested in request['data'])

    Returns:
        True if update succeeds, otherwise False.
    """
    if not isinstance(request, Mapping):
        return False

    headers = request.get("headers")
    if not isinstance(headers, Mapping) or not validate_csrf_token(headers):
        return False

    # Extract user_id from common locations
    user_id: Optional[Union[str, int]] = request.get("user_id")
    if user_id is None:
        user = request.get("user")
        if isinstance(user, Mapping):
            user_id = user.get("id")
    if user_id is None:
        params = request.get("params")
        if isinstance(params, Mapping):
            user_id = params.get("user_id")
    if user_id is None:
        return False

    # Extract new phone number from common locations
    new_phone: Optional[str] = None
    raw_phone = request.get("telephone", request.get("new_phone"))
    if isinstance(raw_phone, (str, int)):
        new_phone = str(raw_phone)

    if not new_phone:
        data = request.get("data")
        if isinstance(data, Mapping):
            raw_phone = data.get("telephone", data.get("new_phone"))
            if isinstance(raw_phone, (str, int)):
                new_phone = str(raw_phone)

    if not new_phone:
        return False

    users = db_users.get("users")
    if not isinstance(users, dict):
        return False

    key = str(user_id)
    user_entry = users.get(key)
    if not isinstance(user_entry, dict):
        return False

    user_entry["telephone"] = new_phone
    return True
