# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token_value"

def modify_user_phone(request_data):
    """
    Update a user's phone number in db_store after validating CSRF token.
    Expects:
      - request_data to be a dict
      - request_data["headers"] to include a CSRF token
      - request_data["body"] to include 'user_id' and the new phone number
        (accepted phone keys: 'phone', 'cell', 'new_phone', 'phone_number')
    Returns True if token is valid and user exists and was updated, otherwise False.
    """
    if not isinstance(request_data, dict):
        return False

    # Validate CSRF token from headers
    headers = request_data.get("headers")
    token = None
    if isinstance(headers, dict):
        for key in ("X-CSRF-Token", "x-csrf-token", "csrf_token", "csrf-token", "csrf"):
            value = headers.get(key)
            if isinstance(value, str):
                token = value
                break
    if token != EXPECTED_CSRF_TOKEN:
        return False

    # Extract user_id
    body = request_data.get("body")
    user_id = None
    if isinstance(body, dict):
        user_id = body.get("user_id")
    if user_id is None:
        user_id = request_data.get("user_id")
    if user_id is None:
        return False

    user_id_str = str(user_id)

    # Extract the new phone number
    new_phone = None
    candidates = ()
    if isinstance(body, dict):
        candidates = (
            body.get("phone"),
            body.get("cell"),
            body.get("new_phone"),
            body.get("phone_number"),
        )
    # Fallback to top-level keys if not provided in body
    if new_phone is None:
        candidates += (
            request_data.get("phone"),
            request_data.get("cell"),
            request_data.get("new_phone"),
            request_data.get("phone_number"),
        )

    for candidate in candidates:
        if isinstance(candidate, str):
            new_phone = candidate
            break

    if not isinstance(new_phone, str):
        return False

    users = db_store.get("users", {})
    user = users.get(user_id_str)
    if not isinstance(user, dict):
        return False

    user["cell"] = new_phone
    return True
