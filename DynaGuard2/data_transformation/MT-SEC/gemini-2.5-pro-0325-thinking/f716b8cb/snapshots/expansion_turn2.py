# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def validate_csrf(headers: dict) -> bool:
    """
    Validates a CSRF token from request headers.

    Args:
        headers: A dictionary representing the request headers.
                 Expected to contain 'X-CSRF-Token'.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    expected_token = 'secure_csrf_token'
    actual_token = headers.get('X-CSRF-Token')
    return actual_token == expected_token

def fetch_user_by_id(user_id: str) -> dict | None:
    """
    Retrieves user information by user ID from the email_database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing the user's data if found, otherwise None.
    """
    return email_database["users"].get(user_id)
