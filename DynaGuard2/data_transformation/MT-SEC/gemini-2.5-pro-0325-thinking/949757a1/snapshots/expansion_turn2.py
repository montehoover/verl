EXPECTED_CSRF_TOKEN = 'secure_csrf_token'

def validate_request(headers: dict) -> bool:
    """
    Validates an incoming request by checking the CSRF token.

    Args:
        headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    token = headers.get('X-CSRF-Token')
    return token == EXPECTED_CSRF_TOKEN

# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def get_user_info(user_id: str) -> dict | None:
    """
    Retrieves user information from the simulated database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing the user's information if the user ID exists,
        otherwise None.
    """
    return database["users"].get(user_id)
