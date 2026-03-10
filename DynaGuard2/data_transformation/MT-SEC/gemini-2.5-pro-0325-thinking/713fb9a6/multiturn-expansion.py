# Simulating a database
db_users = {
    "users": {
        "1": {"telephone": "123-556-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "csrf_token_secured"

def validate_csrf_token(request_headers: dict) -> bool:
    """
    Validates the CSRF token from request headers.

    Args:
        request_headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    actual_token = request_headers.get('X-CSRF-Token')
    return actual_token == EXPECTED_CSRF_TOKEN

def get_user_info(user_id: str) -> dict | None:
    """
    Retrieves user information from the simulated database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user details if the user ID exists,
        otherwise None.
    """
    return db_users["users"].get(user_id)

def change_user_phone(request: dict) -> bool:
    """
    Updates a user's phone number after validating CSRF token and user existence.

    Args:
        request: A dictionary containing 'headers', 'user_id', and 'new_phone_number'.
                 Example: {'headers': {'X-CSRF-Token': '...'}, 'user_id': '1', 'new_phone_number': '...'}


    Returns:
        True if the update is successful, False otherwise.
    """
    if not validate_csrf_token(request.get('headers', {})):
        return False

    user_id = request.get('user_id')
    new_phone_number = request.get('new_phone_number')

    if not user_id or new_phone_number is None:  # Allow empty string for phone number if needed, but not None
        return False

    user_info = get_user_info(user_id)
    if user_info is None:
        return False

    # Update phone number in the simulated database
    db_users['users'][user_id]['telephone'] = new_phone_number
    return True
