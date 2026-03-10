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

# Global set of admin user IDs
# Example: {100, 101} (assuming user IDs 100 and 101 are admins)
ADMIN_USER_IDS = set()

def view_profile_with_access(active_user_id: int, profile_user_id: int):
    """
    Allows admins to view any user's profile, while regular users can only view their own.

    Args:
        active_user_id: The integer ID of the user making the request.
        profile_user_id: The integer ID of the profile being requested.

    Returns:
        The user's profile dictionary if access is allowed, otherwise the string "Unauthorized".
    """
    is_admin = active_user_id in ADMIN_USER_IDS
    
    if is_admin or active_user_id == profile_user_id:
        profile = get_user_profile(profile_user_id)
        return profile if profile is not None else "Profile not found"
    else:
        return "Unauthorized"
