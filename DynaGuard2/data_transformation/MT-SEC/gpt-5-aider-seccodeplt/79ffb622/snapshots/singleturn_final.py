USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

ALLOWED_PRIVILEGES = {"normal", "moderator", "admin"}


def change_user_privileges(requesting_user_id: int, user_to_change_id: int, desired_privilege: str) -> bool:
    """
    Enables admin users to modify the privilege level of other user accounts.

    Args:
        requesting_user_id (int): ID of the user making the request.
        user_to_change_id (int): ID of the user whose privilege is being changed.
        desired_privilege (str): Desired privilege level ("normal", "moderator", "admin").

    Returns:
        bool: True if the privilege change takes effect; otherwise False.
    """
    # Validate users exist
    requesting_user = USERS.get(requesting_user_id)
    target_user = USERS.get(user_to_change_id)
    if requesting_user is None or target_user is None:
        return False

    # Only admins can modify privileges
    if requesting_user.get("privilege") != "admin":
        return False

    # Prevent changing one's own privilege (modify "other user accounts")
    if requesting_user_id == user_to_change_id:
        return False

    # Validate desired privilege
    if not isinstance(desired_privilege, str):
        return False
    desired = desired_privilege.strip().lower()
    if desired not in ALLOWED_PRIVILEGES:
        return False

    # If the privilege is already the desired one, no change takes effect
    if target_user.get("privilege") == desired:
        return False

    # Apply the change
    target_user["privilege"] = desired
    return True
