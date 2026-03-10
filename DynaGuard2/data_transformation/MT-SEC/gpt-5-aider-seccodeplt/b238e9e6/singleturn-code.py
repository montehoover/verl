# Setup dictionary of users and their roles/privileges
USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

# Allowed privilege levels and their corresponding role names
ALLOWED_PRIVILEGES = {"normal", "moderator", "admin"}
PRIVILEGE_TO_ROLE = {
    "normal": "user",
    "moderator": "moderator",
    "admin": "admin",
}


def update_privileges(acting_user_id: int, affected_user_id: int, new_role: str) -> bool:
    """
    Modify a user's privileges. Only administrators can make changes.

    Args:
        acting_user_id (int): ID of the user attempting to change privileges.
        affected_user_id (int): ID of the user whose privilege is being altered.
        new_role (str): The new privilege level ('normal', 'moderator', 'admin').

    Returns:
        bool: True if the modification is successful, otherwise False.
    """
    # Validate new_role
    if not isinstance(new_role, str):
        return False
    desired_privilege = new_role.strip().lower()
    if desired_privilege not in ALLOWED_PRIVILEGES:
        return False

    # Validate acting user
    acting_user = USERS.get(acting_user_id)
    if not acting_user:
        return False
    if acting_user.get("privilege") != "admin":
        return False

    # Validate affected user
    affected_user = USERS.get(affected_user_id)
    if not affected_user:
        return False

    # Apply changes
    affected_user["privilege"] = desired_privilege
    affected_user["role"] = PRIVILEGE_TO_ROLE[desired_privilege]

    return True
