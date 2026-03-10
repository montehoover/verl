USER_PROFILES = {
    "123": {"name": "Alice Wonderland", "email": "alice@example.com"},
    "456": {"name": "Bob The Builder", "email": "bob@example.com"},
    "789": {"name": "Charlie Brown", "email": "charlie@example.com"},
}

def get_user_profile(user_id: str):
    """
    Retrieves a user profile from a local dictionary.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary with the user's name and email if found, otherwise None.
    """
    user_data = USER_PROFILES.get(user_id)
    if user_data:
        return {"name": user_data["name"], "email": user_data["email"]}
    return None

if __name__ == '__main__':
    # Example Usage
    user1 = get_user_profile("123")
    print(f"User 123: {user1}")

    user2 = get_user_profile("456")
    print(f"User 456: {user2}")

    user_non_existent = get_user_profile("999")
    print(f"User 999: {user_non_existent}")

    user_another = get_user_profile("789")
    print(f"User 789: {user_another}")
