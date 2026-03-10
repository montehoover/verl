# Global data structure for user profiles
user_profiles = {
    1: {"name": "Alice", "email": "alice@example.com", "age": 30},
    2: {"name": "Bob", "email": "bob@example.com", "age": 24},
    3: {"name": "Charlie", "email": "charlie@example.com", "age": 35}
}

def view_own_profile(user_id: int):
    """
    Allows a user to view their own profile information.

    Args:
        user_id: The ID of the user.

    Returns:
        The user's profile details if the user exists, otherwise 'User not found'.
    """
    if user_id in user_profiles:
        return user_profiles[user_id]
    else:
        return "User not found"
