# Global data structure for user profiles
# Example: user_profiles = {1: {"name": "Alice", "email": "alice@example.com"}, 2: {"name": "Bob", "email": "bob@example.com"}}
user_profiles = {}

def view_own_profile(user_id: int):
    """
    Allows a user to view their own profile details.

    Args:
        user_id: The ID of the user.

    Returns:
        The profile details if the user exists, otherwise "Profile not found".
    """
    if user_id in user_profiles:
        return user_profiles[user_id]
    else:
        return "Profile not found"
