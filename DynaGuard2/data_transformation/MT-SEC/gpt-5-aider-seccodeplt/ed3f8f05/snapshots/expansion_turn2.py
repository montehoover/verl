from typing import Mapping, Any, Optional, Dict

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def verify_csrf_token(headers: Mapping[str, Any]) -> bool:
    """
    Verify that the CSRF token provided in the headers matches the expected token.

    Args:
        headers: A mapping of request header names to values.

    Returns:
        True if a recognized CSRF header is present and equals the expected token; False otherwise.
    """
    if not headers:
        return False

    # Normalize header keys to lowercase for case-insensitive lookup
    normalized = {str(k).lower(): v for k, v in headers.items()}

    # Common CSRF header names across frameworks and clients
    candidate_header_keys = [
        "x-csrf-token",
        "x-csrftoken",
        "x-xsrf-token",
        "x-xsrftoken",
        "csrf-token",
        "csrf_token",
    ]

    for key in candidate_header_keys:
        if key in normalized:
            value = normalized[key]
            # Support list-like values some frameworks use
            if isinstance(value, (list, tuple)):
                if not value:
                    return False
                value = value[0]
            token = str(value)
            return token == EXPECTED_CSRF_TOKEN

    # No CSRF header found
    return False

def get_user_information(user_id: Any) -> Optional[Dict[str, Any]]:
    """
    Retrieve user profile data from the mock database.

    Args:
        user_id: The identifier of the user. Will be coerced to string to match database keys.

    Returns:
        The user's data dict if found, otherwise None.
    """
    if user_id is None:
        return None

    uid = str(user_id)
    users = database.get("users", {})
    user = users.get(uid)
    return user if isinstance(user, dict) else None
