USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin", "status": "active"},
    2: {"user_id": 2, "role": "user", "privilege": "normal", "status": "active"},
    3: {"user_id": 3, "role": "user", "privilege": "normal", "status": "active"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator", "status": "active"}
}

def change_user_status(current_user_id: int, target_user_id: int, new_status: str) -> bool:
    """
    Allows an admin user to modify the profile status of a target user.

    Args:
        current_user_id: The ID of the user attempting the change.
        target_user_id: The ID of the user whose status is to be changed.
        new_status: The new status to set for the target user.

    Returns:
        True if the status was successfully changed, False otherwise.
    """
    admin_user = USERS.get(current_user_id)
    
    if not admin_user or admin_user.get("privilege") != "admin":
        return False  # Current user is not an admin or does not exist

    target_user = USERS.get(target_user_id)
    
    if not target_user:
        return False  # Target user does not exist

    target_user["status"] = new_status
    return True
