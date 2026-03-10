USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def is_admin(user_id: int) -> bool:
    """
    Checks if a user has admin privileges.

    Args:
        user_id: The ID of the user.

    Returns:
        True if the user has admin privileges, False otherwise.
    """
    user = USERS.get(user_id)
    if user:
        return user.get("role") == "admin"
    return False
