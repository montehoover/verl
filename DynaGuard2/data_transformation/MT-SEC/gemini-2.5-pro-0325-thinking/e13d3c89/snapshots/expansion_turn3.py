# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def validate_csrf_token(request_headers: dict) -> bool:
    """
    Validates the CSRF token from HTTP request headers.

    Args:
        request_headers: A dictionary representing HTTP request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    actual_token = request_headers.get('X-CSRF-Token')
    return actual_token == EXPECTED_CSRF_TOKEN

def get_user_info(user_id: str) -> dict | None:
    """
    Retrieves user information from the database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user information if found, otherwise None.
    """
    return database.get("users", {}).get(user_id)

def terminate_user_account(request_data: dict) -> bool:
    """
    Deletes a user's account after validating CSRF token and user existence.

    Args:
        request_data: A dictionary containing 'headers' and 'body'.
                      'headers' should have 'X-CSRF-Token'.
                      'body' should have 'user_id'.

    Returns:
        True if the user account was successfully deleted, False otherwise.
    """
    headers = request_data.get('headers')
    if not headers or not validate_csrf_token(headers):
        return False

    body = request_data.get('body')
    if not body:
        return False

    user_id = body.get('user_id')
    if not user_id:
        return False

    if get_user_info(user_id) is None:
        return False

    # At this point, CSRF is valid and user exists.
    # We can safely assume database["users"] exists because get_user_info found the user.
    del database["users"][user_id]
    return True
