USER_DATA_STORE = {
    "user123": {"name": "Alice Wonderland", "email": "alice@example.com", "age": 30},
    "user456": {"name": "Bob The Builder", "email": "bob@example.com", "age": 45},
    "user789": {"name": "Charlie Brown", "email": "charlie@example.com", "age": 8},
}

def retrieve_user_profile(user_id: str) -> dict:
    """
    Retrieves the profile information for a given user_id from the USER_DATA_STORE.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        The user's profile data.

    Raises:
        KeyError: If the user_id does not exist in the USER_DATA_STORE.
    """
    if user_id in USER_DATA_STORE:
        return USER_DATA_STORE[user_id]
    else:
        raise KeyError(f"User ID '{user_id}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        profile = retrieve_user_profile("user123")
        print(f"Profile for user123: {profile}")
    except KeyError as e:
        print(e)

    try:
        profile = retrieve_user_profile("user000")
        print(f"Profile for user000: {profile}")
    except KeyError as e:
        print(e)
