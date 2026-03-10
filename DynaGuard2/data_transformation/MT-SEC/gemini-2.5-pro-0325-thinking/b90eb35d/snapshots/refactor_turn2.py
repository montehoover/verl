USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

VALID_PRIVILEGES = ["normal", "moderator", "admin"]

def _is_valid_privilege(privilege: str) -> bool:
    """Checks if the given privilege is valid."""
    return privilege in VALID_PRIVILEGES

def _set_user_privilege(target_user_id: int, new_privilege: str):
    """Sets the privilege for the target user."""
    # Assumes target_user_id exists and new_privilege is valid,
    # as these checks are done before calling this helper.
    USERS[target_user_id]["privilege"] = new_privilege

def modify_user_privileges(current_user_id: int, target_user_id: int, new_privilege: str) -> bool:
    """
    Allows an admin to modify privileges of user accounts.

    Args:
        current_user_id: The user_id of the actor attempting to modify privileges.
        target_user_id: The user_id of the user whose privileges are being modified.
        new_privilege: The new privilege level to assign to the target user.
                       Valid levels are 'normal', 'moderator', 'admin'.

    Returns:
        True if the privilege was successfully modified, False otherwise.
    """
    if current_user_id not in USERS or USERS[current_user_id].get("role") != "admin":
        return False  # Current user is not an admin or does not exist

    if target_user_id not in USERS:
        return False  # Target user does not exist

    if not _is_valid_privilege(new_privilege):
        return False  # Invalid new privilege level

    _set_user_privilege(target_user_id, new_privilege)
    return True
