USER_ROLES = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"}
}


def get_user_role(user_id: int) -> str:
    """
    Retrieve the role for a given user_id.

    Args:
        user_id: The ID of the user as an integer.

    Returns:
        The role as a string.

    Raises:
        TypeError: If user_id is not an int.
        KeyError: If the user_id does not exist in USER_ROLES.
        ValueError: If the role is missing or not a string.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an int")

    user = USER_ROLES.get(user_id)
    if user is None:
        raise KeyError(f"user_id {user_id} not found")

    role = user.get("role")
    if not isinstance(role, str):
        raise ValueError(f"Invalid role for user_id {user_id}")

    return role
