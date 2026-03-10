SYSTEM_USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

USER_PROFILES = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
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


def can_access_profile(initiator_id: int, requested_profile_id: int) -> bool:
    """
    Determine if initiator_id is allowed to access requested_profile_id.

    Rules:
      - 'admin' can access any existing profile.
      - 'user' can access only their own profile.

    Args:
        initiator_id: The ID of the user attempting access.
        requested_profile_id: The ID of the profile being accessed.

    Returns:
        True if access is allowed, False otherwise.

    Raises:
        TypeError: If either ID is not an int.
        ValueError: If either user ID does not exist or has invalid data.
    """
    if not isinstance(initiator_id, int) or not isinstance(requested_profile_id, int):
        raise TypeError("initiator_id and requested_profile_id must be int")

    initiator_role = get_user_role(initiator_id)
    # Validate the requested profile exists (and has valid data)
    get_user_role(requested_profile_id)

    if initiator_role == "admin":
        return True
    if initiator_role == "user":
        return initiator_id == requested_profile_id

    return False


def access_profile_details(initiator_id: int, requested_profile_id: int):
    """
    Return the requested profile details if access is allowed, otherwise 'Unauthorized'.

    Args:
        initiator_id: The ID of the user attempting access.
        requested_profile_id: The ID of the profile being requested.

    Returns:
        Dict with profile details if allowed, otherwise the string 'Unauthorized'.

    Raises:
        TypeError: If either ID is not an int.
        ValueError: If either user ID does not exist or profile is missing.
    """
    # Ensure the profile exists
    profile = USER_PROFILES.get(requested_profile_id)
    if profile is None:
        raise ValueError(f"Profile for user id {requested_profile_id} not found")

    if can_access_profile(initiator_id, requested_profile_id):
        return profile

    return "Unauthorized"
