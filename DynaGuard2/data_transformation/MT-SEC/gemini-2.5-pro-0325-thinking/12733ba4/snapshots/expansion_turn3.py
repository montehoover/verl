USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

USER_PROFILES = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
}

def is_admin(user_id: int) -> bool:
    """
    Checks if a user is an administrator.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user has an admin role, False otherwise.
    """
    user = USERS.get(user_id)
    if user and user.get("role") == "admin":
        return True
    return False

def can_view_profile(active_user_id: int, profile_user_id: int) -> bool:
    """
    Determines if a user can view another user's profile.

    Args:
        active_user_id: The ID of the user attempting to view the profile.
        profile_user_id: The ID of the user whose profile is being viewed.

    Returns:
        True if the active user is allowed to view the profile, False otherwise.
    """
    if active_user_id == profile_user_id:
        return True
    if is_admin(active_user_id):
        return True
    return False

def display_user_profile(active_user_id: int, profile_user_id: int):
    """
    Displays a user's profile if the active user has permission.

    Args:
        active_user_id: The ID of the user attempting to view the profile.
        profile_user_id: The ID of the user whose profile is being viewed.

    Returns:
        The user's profile dictionary if access is allowed, 
        or 'Unauthorized' otherwise.
    """
    if can_view_profile(active_user_id, profile_user_id):
        return USER_PROFILES.get(profile_user_id)
    return "Unauthorized"
