USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def check_user_privilege(current_user_id: int) -> str:
    """
    Checks if the user with the given ID has admin privileges.

    Args:
        current_user_id: The ID of the user to check.

    Returns:
        A message indicating whether admin access is confirmed or denied.
    """
    user = USERS.get(current_user_id)
    if user and user.get("privilege") == "admin":
        return "Admin access confirmed"
    else:
        return "Access denied"
