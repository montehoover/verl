# Mock database
MOCK_DB = {
    "user123": {"email": "user123@example.com", "name": "Alice"},
    "user456": {"email": "user456@example.com", "name": "Bob"},
    "user789": {"email": "user789@example.com", "name": "Charlie"},
}

# Predefined authentication token
VALID_AUTH_TOKEN = "secret_token_123"

def get_user_info(user_id: str, auth_token: str):
    """
    Retrieves user information from a database if the user is authenticated.

    Args:
        user_id: The ID of the user to retrieve.
        auth_token: The authentication token for the user.

    Returns:
        The user's email if the user exists and is authenticated, or None otherwise.
    """
    if auth_token != VALID_AUTH_TOKEN:
        print("Error: Invalid authentication token.")
        return None

    if user_id in MOCK_DB:
        return MOCK_DB[user_id].get("email")
    return None
