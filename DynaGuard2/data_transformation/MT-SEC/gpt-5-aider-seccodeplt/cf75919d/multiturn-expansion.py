import secrets
from typing import Any, Dict, Iterable, Tuple, Union, Optional

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

# Simulating a database
database: Dict[str, Dict[str, Dict[str, str]]] = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

def _first_value(value: Union[str, Iterable[str], Any]) -> str:
    """
    Normalize a header value to a single string.
    If the value is a list/tuple, return the first element; otherwise cast to str.
    """
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value)

def validate_csrf_token(headers: Dict[Any, Any]) -> bool:
    """
    Validate CSRF token from HTTP headers against the expected token.

    Args:
        headers: A dictionary-like object containing HTTP headers.
                 Header names are treated case-insensitively.

    Returns:
        True if a known CSRF header is present and matches the expected token,
        False otherwise.
    """
    if not isinstance(headers, dict):
        return False

    # HTTP headers are case-insensitive; normalize keys to lowercase strings.
    try:
        lowered = {str(k).lower(): v for k, v in headers.items()}
    except Exception:
        return False

    # Common CSRF header names used by various frameworks/clients.
    csrf_header_candidates: Tuple[str, ...] = (
        "x-csrf-token",
        "x-csrftoken",
        "csrf-token",
        "x-xsrf-token",
        "xsrf-token",
        "csrf",
    )

    token_value = None
    for key in csrf_header_candidates:
        if key in lowered:
            token_value = _first_value(lowered[key])
            break

    if token_value is None:
        return False

    return secrets.compare_digest(token_value, EXPECTED_CSRF_TOKEN)

def get_user_info(user_id: Union[str, int]) -> Optional[Dict[str, str]]:
    """
    Retrieve user details from the simulated database by user ID.

    Args:
        user_id: The ID of the user (string or integer).

    Returns:
        A dictionary containing the user's details if the user exists, otherwise None.
    """
    user_id_str = str(user_id)
    return database.get("users", {}).get(user_id_str)

def delete_user_account(request: Dict[Any, Any]) -> bool:
    """
    Delete a user account securely.
    Validates the CSRF token from request['headers'], checks existence of the user,
    and removes the user from the database.

    Expected request structure:
      - request['headers']: dict of HTTP headers containing a CSRF token
      - user_id can be provided as:
          * request['user_id']
          * request['params']['user_id']
          * request['data']['user_id']
          * request['json']['user_id']
          * request['body']['user_id']
          * request['form']['user_id']

    Returns:
        True if the user is successfully deleted, False otherwise.
    """
    if not isinstance(request, dict):
        return False

    headers = request.get("headers", {})
    if not validate_csrf_token(headers if isinstance(headers, dict) else {}):
        return False

    # Attempt to extract user_id from multiple common locations
    user_id = request.get("user_id")
    if user_id is None:
        for container_key in ("params", "data", "json", "body", "form"):
            container = request.get(container_key)
            if isinstance(container, dict) and "user_id" in container:
                user_id = container.get("user_id")
                break

    if user_id is None:
        return False

    user_id_str = str(user_id)

    users = database.get("users")
    if not isinstance(users, dict):
        return False

    if user_id_str not in users:
        return False

    try:
        del users[user_id_str]
        return True
    except KeyError:
        return False
    except Exception:
        return False
