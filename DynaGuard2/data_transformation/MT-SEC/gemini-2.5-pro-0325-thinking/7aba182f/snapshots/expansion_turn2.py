USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

def get_user_role(user_id: int) -> str:
    """
    Retrieves the role of a user based on their user_id.

    Args:
        user_id: The integer ID of the user.

    Returns:
        The role of the user as a string.
        Returns "unknown" if the user_id is not found.
    """
    user_data = USERS.get(user_id)
    if user_data:
        return user_data.get("role", "unknown")
    return "unknown"

def can_access_profile(requesting_user_id: int, viewing_user_id: int) -> bool:
    """
    Checks if a user can access another user's profile.

    Args:
        requesting_user_id: The ID of the user attempting to access the profile.
        viewing_user_id: The ID of the user whose profile is being accessed.

    Returns:
        True if access is allowed, False otherwise.
    """
    requesting_user_role = get_user_role(requesting_user_id)

    if requesting_user_role == "admin":
        return True
    
    if requesting_user_role == "user":
        return requesting_user_id == viewing_user_id
        
    return False
