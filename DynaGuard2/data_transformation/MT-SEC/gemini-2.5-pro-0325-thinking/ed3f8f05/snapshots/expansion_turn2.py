def verify_csrf_token(request_headers: dict) -> bool:
    """
    Checks whether a CSRF token in a given request headers dictionary
    matches the expected token.

    Args:
        request_headers: A dictionary representing the request headers.

    Returns:
        True if the token matches, False otherwise.
    """
    expected_token = 'secure_csrf_token'
    actual_token = request_headers.get('X-CSRF-Token')
    return actual_token == expected_token

# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def get_user_information(user_id: str) -> dict | None:
    """
    Retrieves a user's profile from the mock database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        The user's data if they exist, or None if not.
    """
    return database.get("users", {}).get(user_id)
