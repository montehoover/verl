USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}


def is_admin(user_id: int) -> bool:
    """
    Return True if the given user_id corresponds to a user with the 'admin' role, else False.
    """
    user = USERS.get(user_id)
    if not user:
        return False
    return str(user.get("role", "")).lower() == "admin"
