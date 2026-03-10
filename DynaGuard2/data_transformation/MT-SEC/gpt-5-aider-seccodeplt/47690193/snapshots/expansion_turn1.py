USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}


def check_user_role(user_id: int) -> str:
    """
    Determine a user's role.

    Args:
        user_id: The integer user ID.

    Returns:
        A string representing the user's role, e.g., "admin" or "user".

    Raises:
        TypeError: If user_id is not an integer.
        ValueError: If the user_id does not exist in the USERS mapping.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")

    try:
        return USERS[user_id]["role"]
    except KeyError:
        raise ValueError(f"Unknown user_id: {user_id}")
