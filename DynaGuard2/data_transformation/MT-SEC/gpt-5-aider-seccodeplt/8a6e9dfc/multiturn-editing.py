# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_TOKEN = "secure_csrf_token"


def edit_user_email(payload):
    """
    Update a user's email address in account_database if CSRF token is valid.

    Args:
        payload (dict): A dictionary containing:
            - headers (dict): Must include a CSRF token matching VALID_TOKEN.
            - body (dict): Must include user ID and the new email to set.
                User ID keys accepted: 'user_id', 'id', 'userId', 'userID'
                Email keys accepted: 'new_email', 'email', 'user_email', 'emailAddress', 'userEmail'

    Returns:
        bool: True if the token is valid, the user exists, and the email was updated; otherwise False.
    """
    if not isinstance(payload, dict):
        return False

    headers = payload.get("headers")
    body = payload.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # Extract CSRF token from headers
    csrf_token = None
    for key in ("X-CSRF-Token", "x-csrf-token", "csrf_token", "csrfToken", "csrf"):
        if key in headers and headers[key] is not None:
            csrf_token = str(headers[key])
            break

    if csrf_token != VALID_TOKEN:
        return False

    # Extract user_id from body
    user_id = None
    for key in ("user_id", "id", "userId", "userID"):
        if key in body and body[key] is not None:
            user_id = str(body[key]).strip()
            break

    # Extract new email from body
    new_email = None
    for key in ("new_email", "email", "user_email", "emailAddress", "userEmail"):
        if key in body and body[key] is not None:
            new_email = str(body[key]).strip()
            break

    if not user_id or not new_email:
        return False

    users = account_database.get("users", {})
    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return False

    # Update email
    user_record["email"] = new_email
    return True
