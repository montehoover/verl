USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}


def get_user_info(user_id: int) -> dict:
    """
    Retrieve the user's role and subscription level by user_id.

    Args:
        user_id (int): The user's ID.

    Returns:
        dict: {'role': <str>, 'subscription_level': <str>}

    Raises:
        TypeError: If user_id is not an int.
        ValueError: If user_id is not found.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")
    try:
        user = USERS[user_id]
    except KeyError as exc:
        raise ValueError(f"User with id {user_id} not found") from exc
    return {"role": user["role"], "subscription_level": user["subscription_level"]}
