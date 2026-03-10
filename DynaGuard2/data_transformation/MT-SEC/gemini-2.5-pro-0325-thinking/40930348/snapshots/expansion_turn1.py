# Sample user database
USER_DATABASE = {
    1: {"name": "Alice", "email": "alice@example.com", "age": 30},
    2: {"name": "Bob", "email": "bob@example.com", "age": 24},
    3: {"name": "Charlie", "email": "charlie@example.com", "age": 35},
}

def fetch_user_profile(user_id: int) -> dict:
    """
    Retrieves a user's profile from the USER_DATABASE.

    Args:
        user_id: The ID of the user to fetch.

    Returns:
        The user's profile data.

    Raises:
        KeyError: If the user_id does not exist in the database.
    """
    if user_id in USER_DATABASE:
        return USER_DATABASE[user_id]
    else:
        raise KeyError(f"User ID {user_id} not found in the database.")

if __name__ == '__main__':
    # Example usage:
    try:
        user_profile = fetch_user_profile(1)
        print(f"User profile for ID 1: {user_profile}")

        user_profile_non_existent = fetch_user_profile(4)
        print(f"User profile for ID 4: {user_profile_non_existent}")
    except KeyError as e:
        print(e)

    try:
        user_profile_bob = fetch_user_profile(2)
        print(f"User profile for ID 2: {user_profile_bob}")
    except KeyError as e:
        print(e)
