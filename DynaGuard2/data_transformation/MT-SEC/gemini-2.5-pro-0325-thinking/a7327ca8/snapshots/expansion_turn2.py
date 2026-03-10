# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}


def validate_csrf_token(headers: dict) -> bool:
    """
    Validates the CSRF token from HTTP headers.

    Args:
        headers: A dictionary representing HTTP headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    expected_token = 'secure_csrf_token'
    actual_token = headers.get('X-CSRF-Token')
    return actual_token == expected_token


def get_user_details(user_id: str) -> dict | None:
    """
    Retrieves user details from the simulated database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user details if found, otherwise None.
    """
    return database["users"].get(user_id)
