EXPECTED_CSRF_TOKEN = 'secure_csrf_token'
CSRF_HEADER_NAME = 'X-CSRF-Token'

def validate_request(request_headers: dict) -> bool:
    """
    Validates the request by checking the CSRF token.

    Args:
        request_headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    actual_token = request_headers.get(CSRF_HEADER_NAME)
    return actual_token == EXPECTED_CSRF_TOKEN

# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def get_user_info(user_id: str) -> dict | None:
    """
    Retrieves user information from the database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user information if the user exists,
        None otherwise.
    """
    return database.get("users", {}).get(user_id)
