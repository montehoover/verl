USER_DATA = {
    "user123": {"name": "Alice Wonderland", "email": "alice@example.com", "age": 30},
    "user456": {"name": "Bob The Builder", "email": "bob@example.com", "occupation": "Engineer"},
    "user789": {"name": "Charlie Brown", "email": "charlie@example.com", "preferences": ["comics", "dogs"]},
}

def retrieve_user_profile(user_id: str) -> dict:
    """
    Retrieves the user profile for the given user_id from USER_DATA.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        The user's profile dictionary.

    Raises:
        KeyError: If the user_id is not found in USER_DATA.
    """
    if user_id in USER_DATA:
        return USER_DATA[user_id]
    else:
        raise KeyError(f"User ID '{user_id}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        profile = retrieve_user_profile("user123")
        print(f"Profile for user123: {profile}")

        profile_non_existent = retrieve_user_profile("user000")
        print(f"Profile for user000: {profile_non_existent}")
    except KeyError as e:
        print(e)

    try:
        profile_bob = retrieve_user_profile("user456")
        print(f"Profile for user456: {profile_bob}")
    except KeyError as e:
        print(e)
