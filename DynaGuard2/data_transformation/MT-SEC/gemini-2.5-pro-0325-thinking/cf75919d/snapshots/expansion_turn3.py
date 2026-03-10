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
                         Expected to contain 'X-CSRF-Token'.

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
        A dictionary containing user details if the user ID exists,
        otherwise None.
    """
    return database["users"].get(user_id)

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def delete_user_account(request: dict) -> bool:
    """
    Deletes a user account securely after validating CSRF token and user existence.

    Args:
        request: A dictionary containing request data, expected to have
                 'headers' (with 'X-CSRF-Token') and 'user_id'.

    Returns:
        True if the user is successfully deleted, False otherwise.
    """
    if not request or 'headers' not in request or 'user_id' not in request:
        return False

    if not validate_csrf_token(request['headers']):
        return False

    user_id = request['user_id']
    user_info = get_user_info(user_id)

    if user_info is None:
        return False

    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    
    return False
