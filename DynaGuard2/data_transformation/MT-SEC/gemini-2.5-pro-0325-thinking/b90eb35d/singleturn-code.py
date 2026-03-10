USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

VALID_PRIVILEGES = ["normal", "moderator", "admin"]

def modify_user_privileges(current_user_id: int, target_user_id: int, new_privilege: str) -> bool:
    """
    Allows an admin to modify privileges of user accounts.

    Args:
        current_user_id: The user_id of the actor attempting to modify privileges.
        target_user_id: The user_id of the user whose privileges are being modified.
        new_privilege: The new privilege level to assign to the target user.

    Returns:
        True if the privilege was successfully modified, else False.
    """
    # Check if the current user exists and is an admin
    if current_user_id not in USERS or USERS[current_user_id].get("role") != "admin":
        return False

    # Check if the target user exists
    if target_user_id not in USERS:
        return False

    # Check if the new privilege is valid
    if new_privilege not in VALID_PRIVILEGES:
        return False

    # Modify the privilege
    USERS[target_user_id]["privilege"] = new_privilege
    return True
