USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

VALID_PRIVILEGES = ["normal", "moderator", "admin"]

def change_user_privileges(requesting_user_id: int, user_to_change_id: int, desired_privilege: str) -> bool:
    """
    Allows an admin user to modify the privilege level of another user.

    Args:
        requesting_user_id: The ID of the user making the request.
        user_to_change_id: The ID of the user whose privilege is being changed.
        desired_privilege: The desired privilege level ('normal', 'moderator', 'admin').

    Returns:
        True if the privilege change was successful, False otherwise.
    """
    # Check if the requesting user exists and is an admin
    requesting_user = USERS.get(requesting_user_id)
    if not requesting_user or requesting_user.get("privilege") != "admin":
        return False

    # Check if the user to change exists
    user_to_change = USERS.get(user_to_change_id)
    if not user_to_change:
        return False

    # Check if the desired privilege is valid
    if desired_privilege not in VALID_PRIVILEGES:
        return False

    # Apply the privilege change
    USERS[user_to_change_id]["privilege"] = desired_privilege
    return True
