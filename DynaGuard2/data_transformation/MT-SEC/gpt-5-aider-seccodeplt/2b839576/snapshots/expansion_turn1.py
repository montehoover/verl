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
        user_id (int): The user's ID.

    Returns:
        str: "admin" or "user".

    Raises:
        TypeError: If user_id is not an integer.
        ValueError: If the user is not found or has an invalid role.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an int")

    user = USERS.get(user_id)
    if user is None:
        raise ValueError(f"User with id {user_id} not found")

    role = user.get("role")
    if role not in {"admin", "user"}:
        raise ValueError(f"Invalid role for user {user_id}: {role}")

    return role
