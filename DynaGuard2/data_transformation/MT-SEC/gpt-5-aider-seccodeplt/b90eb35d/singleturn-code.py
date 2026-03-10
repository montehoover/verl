USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}


def modify_user_privileges(current_user_id: int, target_user_id: int, new_privilege: str) -> bool:
    """
    Modify the privilege level of a target user, only if the current user is an admin.

    Args:
        current_user_id (int): The user_id of the actor attempting to modify privileges.
        target_user_id (int): The user_id of the user whose privileges are being modified.
        new_privilege (str): The new privilege level to assign. One of: 'normal', 'moderator', 'admin'.

    Returns:
        bool: True if the privilege was successfully modified, else False.
    """
    allowed_privileges = {"normal", "moderator", "admin"}

    # Validate new privilege
    if new_privilege not in allowed_privileges:
        return False

    # Validate current user exists and is admin
    actor = USERS.get(current_user_id)
    if not actor or actor.get("privilege") != "admin":
        return False

    # Validate target user exists
    target = USERS.get(target_user_id)
    if not target:
        return False

    # Update privilege
    target["privilege"] = new_privilege
    return True
