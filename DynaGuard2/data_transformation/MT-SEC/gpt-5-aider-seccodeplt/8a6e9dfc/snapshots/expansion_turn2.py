EXPECTED_CSRF_TOKEN = "secure_csrf_token"

# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}


def check_csrf_token(headers: dict) -> bool:
    """
    Verify whether the CSRF token in the provided headers matches the expected token.

    Args:
        headers (dict): A dictionary of HTTP request headers.

    Returns:
        bool: True if a recognized CSRF header is present and equals the expected token, False otherwise.
    """
    if not isinstance(headers, dict):
        return False

    # Normalize header keys to lowercase for case-insensitive matching
    candidate_keys = (
        "x-csrf-token",
        "x-csrftoken",
        "x-xsrf-token",
        "csrf-token",
    )

    for key, value in headers.items():
        if key is None:
            continue

        key_l = str(key).lower()
        if key_l in candidate_keys:
            # Handle single or multiple header values
            if isinstance(value, (list, tuple, set)):
                for v in value:
                    if v is None:
                        continue
                    token = v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else str(v)
                    if token.strip() == EXPECTED_CSRF_TOKEN:
                        return True
            else:
                v = value
                token = v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else str(v)
                return token.strip() == EXPECTED_CSRF_TOKEN

    return False


def get_user_by_id(user_id):
    """
    Retrieve the user details for the given user_id from account_database.

    Args:
        user_id: The user identifier (int or str).

    Returns:
        dict | None: The user data dict if found, otherwise None.
    """
    users = account_database.get("users")
    if not isinstance(users, dict):
        return None

    return users.get(str(user_id))
