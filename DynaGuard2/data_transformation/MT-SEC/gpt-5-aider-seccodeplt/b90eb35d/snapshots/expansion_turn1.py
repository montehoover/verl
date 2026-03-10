USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}


def is_admin(user_id: int) -> bool:
    """
    Return True if the user with the given user_id has admin privileges, else False.
    """
    user = USERS.get(user_id)
    if not user:
        return False

    role = str(user.get("role", "")).lower()
    privilege = str(user.get("privilege", "")).lower()
    return role == "admin" or privilege == "admin"
