# Mock database
MOCK_DB = {
    "user123": {"email": "user123@example.com", "name": "Alice"},
    "user456": {"email": "user456@example.com", "name": "Bob"},
    "user789": {"email": "user789@example.com", "name": "Charlie"},
}

def get_user_info(user_id: str):
    """
    Retrieves user information from a database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        The user's email if the user exists in the database, or None otherwise.
    """
    if user_id in MOCK_DB:
        return MOCK_DB[user_id].get("email")
    return None
