USER_PROFILES = {
    "1": {"name": "Alice", "email": "alice@example.com"},
    "2": {"name": "Bob", "email": "bob@example.com"},
    "3": {"name": "Charlie", "email": "charlie@example.com"},
}

def get_user_profile(user_id: str):
    """
    Retrieves a user profile from a local dictionary based on a user ID.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary with the user's name and email if found, otherwise None.
    """
    user_data = USER_PROFILES.get(user_id)
    if user_data:
        return {"name": user_data["name"], "email": user_data["email"]}
    return None
