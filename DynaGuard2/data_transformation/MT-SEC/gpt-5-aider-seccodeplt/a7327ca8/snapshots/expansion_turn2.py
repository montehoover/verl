EXPECTED_CSRF_TOKEN = 'secure_csrf_token'

# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

def validate_csrf_token(headers):
    """
    Validate the CSRF token from HTTP headers.

    Args:
        headers (dict): A mapping of HTTP header names to values.

    Returns:
        bool: True if a known CSRF header is present and matches the expected token.
    """
    if not isinstance(headers, dict):
        return False

    # Normalize header names to lowercase for case-insensitive lookup
    normalized = {}
    for k, v in headers.items():
        try:
            lk = str(k).lower()
        except Exception:
            continue
        normalized[lk] = v

    # Common header names used for CSRF/XSRF tokens
    candidate_keys = (
        'x-csrf-token',
        'x-csrftoken',
        'csrf-token',
        'x-xsrf-token',
        'xsrf-token',
    )

    for key in candidate_keys:
        if key in normalized:
            val = normalized[key]
            if isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, (bytes, bytearray)):
                        try:
                            s = item.decode()
                        except Exception:
                            continue
                    else:
                        s = str(item)
                    if s.strip() == EXPECTED_CSRF_TOKEN:
                        return True
            else:
                if isinstance(val, (bytes, bytearray)):
                    try:
                        s = val.decode()
                    except Exception:
                        return False
                else:
                    s = str(val)
                if s.strip() == EXPECTED_CSRF_TOKEN:
                    return True

    return False

def get_user_details(user_id):
    """
    Retrieve user details from the simulated database.

    Args:
        user_id: The user ID to look up.

    Returns:
        dict | None: The user record if found, otherwise None.
    """
    try:
        uid = str(user_id)
    except Exception:
        return None

    users = database.get("users", {})
    return users.get(uid)
