# Simulating a database
db_users = {
    "users": {
        "1": {"telephone": "123-556-7890"},
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
    expected_token = 'csrf_token_secured'
    actual_token = request_headers.get('X-CSRF-Token')
    return actual_token == expected_token

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
