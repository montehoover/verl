SYSTEM_USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}


def get_user_role(user_id: int) -> str:
    """
    Return the role for the given user_id.

    Args:
        user_id: The user's ID as an integer.

    Returns:
        The user's role as a string.

    Raises:
        TypeError: If user_id is not an int.
        ValueError: If the user_id does not exist or role data is invalid.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an int")

    user = SYSTEM_USERS.get(user_id)
    if user is None:
        raise ValueError(f"User with id {user_id} not found")

    role = user.get("role")
    if not isinstance(role, str):
        raise ValueError(f"Invalid role data for user id {user_id}")

    return role
