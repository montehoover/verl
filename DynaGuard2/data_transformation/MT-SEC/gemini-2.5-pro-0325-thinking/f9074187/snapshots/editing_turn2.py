# Global data structure for user roles
# Example: {user_id: "role"}
USER_ROLES = {
    1: "admin",
    2: "editor",
    3: "viewer",
    4: "admin",
}

def is_admin(user_id: int) -> bool:
    """
    Checks if a given user is an admin.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user is an admin, False otherwise.
    """
    return USER_ROLES.get(user_id) == "admin"

# Global data structure for user subscription levels
# Example: {user_id: "subscription_level"}
USER_SUBSCRIPTIONS = {
    1: "premium",
    2: "basic",
    3: "premium",
    5: "basic", # User 4 has a role but no subscription
}

# Define the hierarchy of access levels
ACCESS_LEVEL_HIERARCHY = {
    "basic": 1,
    "premium": 2,
}

def has_access_level(user_id: int, required_level: str) -> bool:
    """
    Checks if a user's subscription level meets or exceeds the required level.

    Args:
        user_id: The ID of the user.
        required_level: The required subscription level ('basic' or 'premium').

    Returns:
        True if the user's subscription level is sufficient, False otherwise.
    """
    user_level_str = USER_SUBSCRIPTIONS.get(user_id)

    if not user_level_str:
        return False  # User has no subscription or unknown user

    user_level_val = ACCESS_LEVEL_HIERARCHY.get(user_level_str)
    required_level_val = ACCESS_LEVEL_HIERARCHY.get(required_level)

    if user_level_val is None or required_level_val is None:
        # This case handles invalid required_level strings or unexpected data in USER_SUBSCRIPTIONS
        return False 

    return user_level_val >= required_level_val
