from typing import Optional

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def validate_request(headers: dict) -> bool:
    """
    Validate request headers for a CSRF token.

    Args:
        headers (dict): Dictionary of request headers.

    Returns:
        bool: True if a CSRF token is present and matches the expected token,
              False otherwise.
    """
    if not isinstance(headers, dict):
        return False

    # Normalize header keys to lowercase for case-insensitive matching
    normalized = {str(k).lower(): v for k, v in headers.items()}

    # Common header keys used for CSRF tokens across frameworks
    possible_keys = [
        "x-csrf-token",
        "x-xsrf-token",
        "x-csrftoken",
        "csrf-token",
        "csrf",
        "csrf_token",
        "x-csrf",
    ]

    token = None
    for key in possible_keys:
        if key in normalized:
            value = normalized[key]
            if isinstance(value, (list, tuple)):
                token = value[0] if value else None
            else:
                token = value
            break

    if token is None:
        return False

    if not isinstance(token, str):
        token = str(token)

    token = token.strip()
    return token == EXPECTED_CSRF_TOKEN


def get_user_info(user_id: object) -> Optional[dict]:
    """
    Retrieve user information from the simulated database by user ID.

    Args:
        user_id (object): The user's ID (string or any type convertible to str).

    Returns:
        dict | None: The user's information if found, otherwise None.
    """
    users = database.get("users", {})
    return users.get(str(user_id))
