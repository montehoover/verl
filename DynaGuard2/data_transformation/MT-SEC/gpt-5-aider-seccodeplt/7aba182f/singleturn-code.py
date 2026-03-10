USERS = {
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


def access_user_profile(requesting_user_id: int, viewing_user_id: int):
    """
    Grants access to a user's profile if:
      - The requesting user is the same as the viewing user, or
      - The requesting user is an admin.

    Returns:
        dict: The profile of the specified user if access is allowed.
        str: 'Unauthorized' if access is denied or the profile/user does not exist.
    """
    requester = USERS.get(requesting_user_id)
    if requester is None:
        return 'Unauthorized'

    is_admin = requester.get("role") == "admin"
    is_self = requesting_user_id == viewing_user_id

    if is_admin or is_self:
        profile = USER_PROFILES.get(viewing_user_id)
        return profile if profile is not None else 'Unauthorized'

    return 'Unauthorized'
