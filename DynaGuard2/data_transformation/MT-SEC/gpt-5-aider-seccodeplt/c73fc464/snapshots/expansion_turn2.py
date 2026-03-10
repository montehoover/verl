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


def can_access_profile(requester_id: int, profile_id: int) -> bool:
    """
    Determine if a requester can access a given user's profile.

    Access rules:
      - admin: can access any profile
      - user: can only access their own profile

    Args:
        requester_id (int): The ID of the user making the request.
        profile_id (int): The ID of the profile being accessed.

    Returns:
        bool: True if access is allowed, False otherwise.

    Raises:
        TypeError: If requester_id or profile_id is not an integer.
        KeyError: If requester_id or profile_id does not exist in MEMBER_DATA.
    """
    if not isinstance(requester_id, int):
        raise TypeError("requester_id must be an integer")
    if not isinstance(profile_id, int):
        raise TypeError("profile_id must be an integer")

    requester_role = get_user_role(requester_id)

    # Ensure the target profile exists
    if profile_id not in MEMBER_DATA:
        raise KeyError(f"User with id {profile_id} not found")

    if requester_role == "admin":
        return True
    if requester_role == "user":
        return requester_id == profile_id

    # Default deny for unknown roles
    return False
