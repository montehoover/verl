USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}


def _user_exists(user_id: int) -> bool:
    """Check if a user exists in the system."""
    return user_id in USERS


def _has_admin_privileges(user_id: int) -> bool:
    """Check if a user has admin privileges."""
    if not _user_exists(user_id):
        return False
    return USERS[user_id]["privilege"] == "admin"


def _is_valid_privilege_level(privilege_level: str) -> bool:
    """Validate if the privilege level is allowed."""
    valid_privilege_levels = ["normal", "moderator", "admin"]
    return privilege_level in valid_privilege_levels


def _get_role_for_privilege(privilege_level: str) -> str:
    """Map privilege level to corresponding role."""
    privilege_to_role_mapping = {
        "admin": "admin",
        "moderator": "moderator",
        "normal": "user"
    }
    return privilege_to_role_mapping.get(privilege_level, "user")


def _update_user_privilege_and_role(user_id: int, new_privilege: str) -> None:
    """Update user's privilege and corresponding role."""
    USERS[user_id]["privilege"] = new_privilege
    USERS[user_id]["role"] = _get_role_for_privilege(new_privilege)


def update_privileges(acting_user_id: int, affected_user_id: int, new_role: str) -> bool:
    """
    Update a user's privileges if the acting user has admin rights.
    
    Args:
        acting_user_id: The ID of the user attempting to change privileges
        affected_user_id: The ID of the user whose privilege is being altered
        new_role: The new privilege level designated for that user
        
    Returns:
        True if the privilege modification is successful, otherwise False
    """
    if not _has_admin_privileges(acting_user_id):
        return False
    
    if not _user_exists(affected_user_id):
        return False
    
    if not _is_valid_privilege_level(new_role):
        return False
    
    _update_user_privilege_and_role(affected_user_id, new_role)
    
    return True
