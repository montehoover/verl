USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}


def get_user_info(user_id: int):
    """
    Retrieve user info for the given user_id.

    Args:
        user_id (int): The user's ID.

    Returns:
        dict | None: Dictionary with keys 'role' and 'privilege' if user exists, otherwise None.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")

    user = USERS.get(user_id)
    if user is None:
        return None

    return {"role": user["role"], "privilege": user["privilege"]}
