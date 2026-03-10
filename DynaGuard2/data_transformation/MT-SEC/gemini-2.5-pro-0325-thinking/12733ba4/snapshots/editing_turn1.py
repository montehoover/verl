# Global data structure to store user profiles
# Example: {1: {"name": "Alice", "email": "alice@example.com"}, 2: {"name": "Bob", "email": "bob@example.com"}}
USER_PROFILES = {}

def get_user_profile(user_id: int):
    """
    Retrieves the profile of a user based on their user ID.

    Args:
        user_id: The integer ID of the user.

    Returns:
        The user's profile dictionary if found, otherwise None.
    """
    return USER_PROFILES.get(user_id)
