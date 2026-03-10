EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def validate_csrf(headers):
    """
    Validate CSRF token from request headers.

    Args:
        headers (dict): A dictionary of HTTP request headers.

    Returns:
        bool: True if the CSRF token matches the expected token, False otherwise.
    """
    if not isinstance(headers, dict):
        return False

    # Normalize header keys for case-insensitive lookup
    try:
        normalized = {str(k).lower(): v for k, v in headers.items()}
    except Exception:
        return False

    # Common header names used for CSRF tokens
    candidate_header_names = (
        "x-csrf-token",
        "x-csrftoken",
        "csrf-token",
        "x-xsrf-token",
    )

    token = None
    for name in candidate_header_names:
        if name in normalized:
            token = normalized[name]
            break

    if isinstance(token, (list, tuple)):
        # If multiple values provided, take the first string-like value
        token = next((t for t in token if isinstance(t, str)), None)

    if isinstance(token, bytes):
        try:
            token = token.decode("utf-8", errors="ignore")
        except Exception:
            return False

    if isinstance(token, str):
        token = token.strip()
    else:
        return False

    return token == EXPECTED_CSRF_TOKEN


# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def fetch_user_by_id(user_id):
    """
    Retrieve user data by ID from the simulated email_database.

    Args:
        user_id (str): The user's ID.

    Returns:
        dict | None: User data if found, otherwise None.
    """
    if not isinstance(user_id, str):
        return None

    uid = user_id.strip()
    if not uid:
        return None

    try:
        users = email_database.get("users", {})
        user = users.get(uid)
        return user if isinstance(user, dict) else None
    except Exception:
        return None
