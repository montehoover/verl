# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def check_user_exists(user_id: str) -> bool:
    """
    Return True if the given user_id exists in the database, otherwise False.
    """
    users = database.get("users", {})
    return user_id in users

def validate_csrf_token(headers: dict) -> bool:
    """
    Return True if the CSRF token contained in headers matches EXPECTED_CSRF_TOKEN, otherwise False.
    The check is case-insensitive for header names and supports a few common CSRF header keys.
    """
    if not isinstance(headers, dict):
        return False

    # Normalize headers to lowercase keys for case-insensitive lookup
    normalized = {}
    for k, v in headers.items():
        try:
            key_lower = k.lower()
        except AttributeError:
            continue
        normalized[key_lower] = v

    token_keys = (
        "x-csrf-token",
        "x-csrftoken",
        "csrf-token",
        "csrf_token",
        "x-xsrf-token",
    )

    token_value = None
    for key in token_keys:
        if key in normalized:
            token_value = normalized[key]
            break

    if token_value is None:
        return False

    # If header value is a list (some frameworks), use the first element
    if isinstance(token_value, (list, tuple)):
        token_value = token_value[0] if token_value else None

    if token_value is None:
        return False

    return str(token_value) == EXPECTED_CSRF_TOKEN
