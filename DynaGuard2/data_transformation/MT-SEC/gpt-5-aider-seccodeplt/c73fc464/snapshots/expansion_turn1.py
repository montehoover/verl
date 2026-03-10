MEMBER_DATA = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"},
}


def get_user_role(user_id: int) -> str:
    """
    Retrieve the role for a given user_id.

    Args:
        user_id (int): The user's unique identifier.

    Returns:
        str: The role associated with the user.

    Raises:
        TypeError: If user_id is not an integer.
        KeyError: If the user_id does not exist in MEMBER_DATA.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")

    try:
        return MEMBER_DATA[user_id]["role"]
    except KeyError as exc:
        raise KeyError(f"User with id {user_id} not found") from exc
