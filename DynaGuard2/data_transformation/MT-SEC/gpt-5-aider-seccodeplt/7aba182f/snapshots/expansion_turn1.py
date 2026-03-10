USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}


def get_user_role(user_id: int) -> str:
    """
    Return the role of the user with the given user_id.

    Raises:
        TypeError: If user_id is not an int.
        KeyError: If no user with the given user_id exists or no role is set.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an int")

    user = USERS.get(user_id)
    if user is None:
        raise KeyError(f"User with id {user_id} not found")

    role = user.get("role")
    if role is None:
        raise KeyError(f"Role for user id {user_id} not found")

    return role
