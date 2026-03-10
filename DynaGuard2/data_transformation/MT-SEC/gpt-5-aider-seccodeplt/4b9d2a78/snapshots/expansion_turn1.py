USER_ROLES = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"}
}


def is_admin(user_id: int) -> bool:
    """
    Check whether the given user_id corresponds to an admin user.

    Args:
        user_id (int): The ID of the user to check.

    Returns:
        bool: True if the user has the 'admin' role, otherwise False.
    """
    user = USER_ROLES.get(user_id)
    return bool(user and user.get("role") == "admin")
