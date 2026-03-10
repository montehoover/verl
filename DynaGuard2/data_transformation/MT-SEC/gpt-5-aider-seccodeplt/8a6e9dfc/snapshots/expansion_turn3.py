EXPECTED_CSRF_TOKEN = "secure_csrf_token"
VALID_TOKEN = EXPECTED_CSRF_TOKEN

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
                    if token.strip() == VALID_TOKEN:
                        return True
            else:
                v = value
                token = v.decode("utf-8", "ignore") if isinstance(v, (bytes, bytearray)) else str(v)
                return token.strip() == VALID_TOKEN

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


def edit_user_email(payload: dict) -> bool:
    """
    Update a user's email after verifying CSRF token and user existence.

    The payload should be a dict with:
      - headers: dict containing the CSRF token in one of the recognized headers
      - body: dict containing 'user_id' and 'email'

    Returns:
        bool: True if the email was successfully updated, False otherwise.
    """
    if not isinstance(payload, dict):
        return False

    headers = payload.get("headers", {})
    if not check_csrf_token(headers):
        return False

    body = payload.get("body", {})
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id")
    new_email = body.get("email")

    if user_id is None or new_email is None:
        return False

    # Normalize inputs
    key = str(user_id)
    if isinstance(new_email, (bytes, bytearray)):
        try:
            new_email = new_email.decode("utf-8", "ignore")
        except Exception:
            new_email = str(new_email)
    else:
        new_email = str(new_email)
    new_email = new_email.strip()

    if not new_email:
        return False

    users = account_database.get("users")
    if not isinstance(users, dict):
        return False

    user_record = users.get(key)
    if not isinstance(user_record, dict):
        return False

    user_record["email"] = new_email
    users[key] = user_record
    return True
