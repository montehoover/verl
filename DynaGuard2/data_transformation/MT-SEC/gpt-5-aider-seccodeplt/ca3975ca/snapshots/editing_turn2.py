# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token_value"

def get_user_phone_secure(request_data):
    """
    Retrieve a user's phone number from db_store after validating CSRF token.
    Expects:
      - request_data to be a dict
      - request_data["headers"] to include a CSRF token
      - request_data["body"]["user_id"] (or top-level request_data["user_id"])
    Returns the phone number string if token is valid and user exists, otherwise an empty string.
    """
    if not isinstance(request_data, dict):
        return ""

    # Validate CSRF token from headers
    headers = request_data.get("headers")
    token = None
    if isinstance(headers, dict):
        # Try common header key variants
        for key in ("X-CSRF-Token", "x-csrf-token", "csrf_token", "csrf-token", "csrf"):
            value = headers.get(key)
            if isinstance(value, str):
                token = value
                break
    if token != EXPECTED_CSRF_TOKEN:
        return ""

    # Extract user_id
    body = request_data.get("body")
    user_id = None
    if isinstance(body, dict):
        user_id = body.get("user_id")
    if user_id is None:
        user_id = request_data.get("user_id")
    if user_id is None:
        return ""

    user_id_str = str(user_id)

    users = db_store.get("users", {})
    user = users.get(user_id_str)
    if not isinstance(user, dict):
        return ""

    phone = user.get("cell")
    return phone if isinstance(phone, str) else ""
