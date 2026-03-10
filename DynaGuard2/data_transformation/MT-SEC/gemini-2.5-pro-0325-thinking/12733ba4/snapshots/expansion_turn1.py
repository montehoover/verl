USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

def is_admin(user_id: int) -> bool:
    """
    Checks if a user is an administrator.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user has an admin role, False otherwise.
    """
    user = USERS.get(user_id)
    if user and user.get("role") == "admin":
        return True
    return False
