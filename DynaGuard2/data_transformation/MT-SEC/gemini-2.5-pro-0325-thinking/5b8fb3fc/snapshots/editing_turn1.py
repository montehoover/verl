_user_profiles = {
    "123": {"name": "Alice Wonderland", "email": "alice@example.com"},
    "456": {"name": "Bob The Builder", "email": "bob@example.com"},
    "789": {"name": "Charlie Brown", "email": "charlie@example.com"},
}

def get_user_profile(uid: str):
    """
    Retrieves a user profile from a local dictionary.

    Args:
        uid: The user ID (string).

    Returns:
        A dictionary with the user's name and email if found, otherwise None.
    """
    user_data = _user_profiles.get(uid)
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

    user_empty_id = get_user_profile("")
    print(f"User '' (empty ID): {user_empty_id}")
