USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

VALID_PRIVILEGES = {"normal", "moderator", "admin"}


def update_privileges(acting_user_id: int, affected_user_id: int, new_role: str) -> bool:
    """
    Update the privilege (and role) of a user, if the acting user is an admin.

    Args:
        acting_user_id (int): ID of the user attempting to change privileges.
        affected_user_id (int): ID of the user whose privilege is being updated.
        new_role (str): New privilege level ('normal', 'moderator', 'admin').

    Returns:
        bool: True if the privilege update is successful, otherwise False.
    """
    # Validate acting user exists
    acting_user = USERS.get(acting_user_id)
    if not acting_user:
        return False

    # Only admins can modify privileges
    if acting_user.get("privilege") != "admin":
        return False

    # Validate affected user exists
    affected_user = USERS.get(affected_user_id)
    if not affected_user:
        return False

    # Validate new role is allowed
    if new_role not in VALID_PRIVILEGES:
        return False

    # Apply the update (keep 'role' and 'privilege' in sync)
    affected_user["role"] = new_role
    affected_user["privilege"] = new_role
    return True
