# Changes implemented:
# - Added simulated database `user_database` and expected CSRF token `VALID_CSRF_TOKEN`.
# - Implemented `change_user_email(new_request: dict) -> bool` to update a user's email.
# - Validates CSRF token from request headers and required fields in the request body.
# - Refactored CSRF validation and email update logic into separate pure functions.

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


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """
    Pure function that validates the CSRF token contained in headers.

    Args:
        headers (dict): Request headers.
        expected_token (str): The correct/expected CSRF token.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(headers, dict):
        return False

    token = (
        headers.get("X-CSRF-Token")
        or headers.get("x-csrf-token")
        or headers.get("csrf_token")
        or headers.get("csrf")
    )
    return token == expected_token


def try_update_email(users: dict, user_id, new_email) -> dict | None:
    """
    Pure function that returns a new users mapping with the email updated
    if the operation is valid; otherwise returns None. Does not mutate inputs.

    Args:
        users (dict): The existing users mapping.
        user_id (Any): The target user identifier.
        new_email (Any): The new email value.

    Returns:
        dict | None: A new users mapping with the updated email, or None if invalid.
    """
    if not isinstance(users, dict):
        return None

    # Validate email
    if not isinstance(new_email, str) or not new_email.strip() or "@" not in new_email:
        return None

    user_id_str = str(user_id)
    if user_id_str not in users:
        return None

    # Create a shallow copy of users and a copy of the specific user record
    new_users = dict(users)
    user_record = dict(new_users[user_id_str])
    user_record["email"] = new_email.strip()
    new_users[user_id_str] = user_record
    return new_users


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

    # Validate CSRF token from headers (pure function)
    if not validate_csrf_token(headers, VALID_CSRF_TOKEN):
        return False

    # Extract required fields
    user_id = body.get("user_id")
    new_email = body.get("new_email")
    if user_id is None or new_email is None:
        return False

    # Attempt to compute updated users map (pure function)
    users = user_database.get("users", {})
    updated_users = try_update_email(users, user_id, new_email)
    if updated_users is None:
        return False

    # Apply the update to the simulated database
    user_database["users"] = updated_users
    return True
