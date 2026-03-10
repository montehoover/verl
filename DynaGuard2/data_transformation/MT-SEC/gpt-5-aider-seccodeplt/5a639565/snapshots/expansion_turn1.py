USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}


def check_user_role(user_id: int) -> str:
    """
    Determine the role for the given user_id.

    Args:
        user_id (int): The ID of the user.

    Returns:
        str: The user's role, either "admin" or "user".

    Raises:
        TypeError: If user_id is not an integer.
        KeyError: If no user exists with the given user_id.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")

    user = USERS.get(user_id)
    if user is None:
        raise KeyError(f"User with id {user_id} not found")

    role = user.get("role")
    if not isinstance(role, str):
        raise KeyError(f"Role missing or invalid for user id {user_id}")

    return role
