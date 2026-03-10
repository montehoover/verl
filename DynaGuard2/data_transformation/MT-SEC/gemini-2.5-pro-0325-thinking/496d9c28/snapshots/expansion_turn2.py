# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
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
    expected_token = 'secure_csrf_token'
    # Common header name for CSRF tokens is 'X-CSRF-Token'
    # or 'X-XSRF-TOKEN' depending on the framework.
    # We'll use 'X-CSRF-Token' for this example.
    actual_token = request_headers.get('X-CSRF-Token')

    return actual_token == expected_token

def get_user_info(user_id: str) -> dict | None:
    """
    Retrieves user information from the simulated database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user details if found, otherwise None.
    """
    return user_database["users"].get(user_id)
