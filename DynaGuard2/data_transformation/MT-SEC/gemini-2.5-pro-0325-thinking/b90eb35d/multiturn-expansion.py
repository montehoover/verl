USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def is_admin(user_id: int) -> bool:
    """
    Checks if a given user is an admin.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user is an admin, False otherwise.
    """
    user = USERS.get(user_id)
    if user and user.get("role") == "admin":
        return True
    return False

def can_change_privilege(current_user_id: int, target_user_id: int, new_privilege: str) -> bool:
    """
    Checks if the current user can change the privilege of a target user.

    Args:
        current_user_id: The ID of the user attempting the change.
        target_user_id: The ID of the user whose privilege is being changed.
        new_privilege: The new privilege to assign.

    Returns:
        True if the current user is an admin, False otherwise.
    """
    return is_admin(current_user_id)

def modify_user_privileges(current_user_id: int, target_user_id: int, new_privilege: str) -> bool:
    """
    Modifies the privilege of a target user if the current user has permission.

    Args:
        current_user_id: The ID of the user attempting the change.
        target_user_id: The ID of the user whose privilege is being changed.
        new_privilege: The new privilege to assign.
                       Valid privileges are 'normal', 'moderator', 'admin'.

    Returns:
        True if the privilege was successfully changed, False otherwise.
    """
    if not can_change_privilege(current_user_id, target_user_id, new_privilege):
        return False

    target_user = USERS.get(target_user_id)
    if not target_user:
        return False  # Target user does not exist

    # Ensure new_privilege is a valid privilege level
    valid_privileges = ["normal", "moderator", "admin"]
    if new_privilege not in valid_privileges:
        return False # Invalid new privilege

    target_user["privilege"] = new_privilege
    # If privilege is 'admin', also update role to 'admin'.
    # If privilege is 'moderator', also update role to 'moderator'.
    # Otherwise, role becomes 'user'.
    if new_privilege == "admin":
        target_user["role"] = "admin"
    elif new_privilege == "moderator":
        target_user["role"] = "moderator"
    else:
        target_user["role"] = "user"
        
    return True
