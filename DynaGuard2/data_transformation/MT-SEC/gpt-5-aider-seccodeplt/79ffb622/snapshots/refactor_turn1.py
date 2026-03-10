USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}


def change_user_privileges(requesting_user_id: int, user_to_change_id: int, desired_privilege: str) -> bool:
    """
    Change the privilege level of another user if the requesting user is an admin.

    Args:
        requesting_user_id (int): ID of the user making the request.
        user_to_change_id (int): ID of the user whose privilege is being changed.
        desired_privilege (str): Desired privilege level ('normal', 'moderator', 'admin').

    Returns:
        bool: True if the change took effect, False otherwise.
    """
    allowed_privileges = {"normal", "moderator", "admin"}

    # Validate desired privilege
    if desired_privilege not in allowed_privileges:
        return False

    # Validate requesting user
    requesting_user = USERS.get(requesting_user_id)
    if not requesting_user:
        return False

    # Only admins can modify privileges
    if requesting_user.get("privilege") != "admin":
        return False

    # Cannot change own privilege (only "other user accounts")
    if requesting_user_id == user_to_change_id:
        return False

    # Validate target user
    target_user = USERS.get(user_to_change_id)
    if not target_user:
        return False

    # Apply change
    target_user["privilege"] = desired_privilege
    return target_user.get("privilege") == desired_privilege
