SYSTEM_USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

def get_user_role(user_id: int) -> str | None:
    """
    Retrieves the role of a user based on their user_id.

    Args:
        user_id: The ID of the user.

    Returns:
        The role of the user as a string, or None if the user is not found.
    """
    user_data = SYSTEM_USERS.get(user_id)
    if user_data:
        return user_data.get("role")
    return None

def can_access_profile(initiator_id: int, requested_profile_id: int) -> bool:
    """
    Checks if a user can access another user's profile.

    Args:
        initiator_id: The ID of the user attempting to access the profile.
        requested_profile_id: The ID of the profile being accessed.

    Returns:
        True if access is allowed, False otherwise.
    """
    initiator_role = get_user_role(initiator_id)

    if initiator_role == "admin":
        # Admins can access any profile, provided the requested profile exists
        return get_user_role(requested_profile_id) is not None
    elif initiator_role == "user":
        # Users can only access their own profile
        return initiator_id == requested_profile_id
    
    # If initiator_role is None (user not found) or any other role, deny access
    return False
