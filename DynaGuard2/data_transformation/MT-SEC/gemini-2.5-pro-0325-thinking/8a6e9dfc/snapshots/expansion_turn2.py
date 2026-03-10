# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def check_csrf_token(headers: dict) -> bool:
    """
    Verifies the CSRF token from request headers.

    Args:
        headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    expected_token = 'secure_csrf_token'
    actual_token = headers.get('X-CSRF-Token')
    return actual_token == expected_token

def get_user_by_id(user_id: str) -> dict | None:
    """
    Retrieves user details from the account_database by user ID.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user data if found, otherwise None.
    """
    return account_database.get("users", {}).get(user_id)
