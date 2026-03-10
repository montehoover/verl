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


def can_access_profile(requesting_user_id: int, viewing_user_id: int) -> bool:
    """
    Determine whether the requesting user can access the viewing user's profile.

    Rules:
    - Admins can access any user's profile.
    - Users can access only their own profile.

    Raises:
        TypeError: If either user id is not an int.
        KeyError: If either user id does not exist.
    """
    if not isinstance(requesting_user_id, int) or not isinstance(viewing_user_id, int):
        raise TypeError("requesting_user_id and viewing_user_id must be ints")

    requesting_role = get_user_role(requesting_user_id)
    # Validate that the viewing user exists
    get_user_role(viewing_user_id)

    if requesting_role == "admin":
        return True

    return requesting_user_id == viewing_user_id
