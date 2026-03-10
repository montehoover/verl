USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def is_admin(user_id: int) -> bool:
    """
    Checks if a user has admin privileges.

    Args:
        user_id: The ID of the user.

    Returns:
        True if the user has admin privileges, False otherwise.
    """
    user = USERS.get(user_id)
    if user:
        return user.get("role") == "admin"
    return False

def can_modify_privilege(acting_user_id: int, affected_user_id: int) -> bool:
    """
    Determines if a user is permitted to change another user's privilege level.

    Args:
        acting_user_id: The ID of the user performing the action.
        affected_user_id: The ID of the user whose privilege is being changed.

    Returns:
        True if the acting user has the authority to change the affected user's privilege, False otherwise.
    """
    # Only admins can modify privileges.
    # In a more complex system, you might have different rules,
    # e.g., admins can't modify other admins, or only super-admins can modify admins.
    # For now, any admin can modify any user's privilege.
    return is_admin(acting_user_id)
