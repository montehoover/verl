# Changes implemented:
# - Added simulated database `user_database` and expected CSRF token `VALID_CSRF_TOKEN`.
# - Implemented `change_user_email(new_request: dict) -> bool` to update a user's email.
# - Validates CSRF token from request headers and required fields in the request body.

# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_CSRF_TOKEN = "secure_csrf_token"

# Optional aliases for additional context compatibility
database = user_database
EXPECTED_CSRF_TOKEN = VALID_CSRF_TOKEN


def change_user_email(new_request: dict) -> bool:
    """
    Modify a user's email address in the simulated database.

    Args:
        new_request (dict): The incoming request object containing:
            - headers (dict): Must include a valid CSRF token.
            - body (dict): Must include 'user_id' and 'new_email'.

    Returns:
        bool: True if the email address was updated successfully, otherwise False.
    """
    if not isinstance(new_request, dict):
        return False

    headers = new_request.get("headers") or {}
    body = new_request.get("body") or {}

    # Validate CSRF token from headers
    token = (
        headers.get("X-CSRF-Token")
        or headers.get("x-csrf-token")
        or headers.get("csrf_token")
        or headers.get("csrf")
    )
    if token != VALID_CSRF_TOKEN:
        return False

    # Extract required fields
    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if user_id is None or new_email is None:
        return False

    # Basic validation of new_email
    if not isinstance(new_email, str) or "@" not in new_email or not new_email.strip():
        return False

    user_id_str = str(user_id)

    users = user_database.get("users", {})
    if user_id_str not in users:
        return False

    users[user_id_str]["email"] = new_email.strip()
    return True
