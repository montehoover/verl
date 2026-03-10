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


def modify_user_email_address(user_request):
    """
    Update a user's email address if the CSRF token is valid.

    Args:
        user_request (dict): Expected to contain:
            - headers (dict): HTTP headers including CSRF token.
            - body (dict): Contains a user identifier and the new email, e.g.:
                { "id": "1", "email": "new.email@example.com" }

    Returns:
        bool: True if update succeeded, False otherwise.
    """
    # Validate request structure
    if not isinstance(user_request, dict):
        return False

    headers = user_request.get("headers", {})
    if not validate_csrf(headers):
        return False

    body = user_request.get("body")
    if not isinstance(body, dict):
        return False

    # Extract user ID (support a few common key names)
    uid = None
    for key in ("user_id", "id", "uid"):
        if key in body:
            uid = body.get(key)
            break

    if isinstance(uid, bytes):
        try:
            uid = uid.decode("utf-8", errors="ignore")
        except Exception:
            return False

    if not isinstance(uid, str):
        return False

    uid = uid.strip()
    if not uid:
        return False

    # Extract new email (support a few common key names)
    new_email = None
    for key in ("email", "new_email", "newEmail"):
        if key in body:
            new_email = body.get(key)
            break

    if isinstance(new_email, bytes):
        try:
            new_email = new_email.decode("utf-8", errors="ignore")
        except Exception:
            return False

    if not isinstance(new_email, str):
        return False

    new_email = new_email.strip()
    if not new_email:
        return False

    # Minimal email validation
    if "@" not in new_email or new_email.startswith("@") or new_email.endswith("@"):
        return False

    # Check that the user exists and update
    user = fetch_user_by_id(uid)
    if user is None:
        return False

    try:
        users = email_database.get("users")
        if not isinstance(users, dict):
            return False
        if uid not in users or not isinstance(users[uid], dict):
            return False

        users[uid]["email"] = new_email
        return True
    except Exception:
        return False
