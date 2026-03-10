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
    expected_token = 'secure_csrf_token'
    actual_token = request_headers.get('X-CSRF-Token')
    return actual_token == expected_token

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
