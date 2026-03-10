EXPECTED_CSRF_TOKEN = 'secure_csrf_token'
CSRF_HEADER_NAME = 'X-CSRF-Token'

# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

def validate_csrf_token(request_headers: dict) -> bool:
    """
    Validates the CSRF token from request headers.

    Args:
        request_headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    actual_token = request_headers.get(CSRF_HEADER_NAME)
    return actual_token == EXPECTED_CSRF_TOKEN

def get_user_info(user_id: str) -> dict | None:
    """
    Retrieves user information from the database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user details if the user exists, None otherwise.
    """
    return database["users"].get(user_id)

def remove_user_profile(request: dict) -> bool:
    """
    Removes a user's profile from the system.

    Args:
        request: A dictionary containing request data, including 'headers'
                 for CSRF validation and 'user_id' for identifying the user.

    Returns:
        True if the user profile was successfully removed, False otherwise.
    """
    if not validate_csrf_token(request.get("headers", {})):
        return False

    user_id = request.get("user_id")
    if not user_id:
        return False

    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    return False
