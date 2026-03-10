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


def display_user_profile(active_user_id: int, profile_user_id: int):
    """
    Returns the profile of the specified user if access is allowed.
    Access is allowed if:
      - The active user is requesting their own profile, or
      - The active user has the 'admin' role.

    Otherwise returns the string 'Unauthorized'.
    """
    active_user = USERS.get(active_user_id)
    if not active_user:
        return "Unauthorized"

    is_admin = active_user.get("role") == "admin"
    is_self = active_user_id == profile_user_id

    if not (is_admin or is_self):
        return "Unauthorized"

    profile = USER_PROFILES.get(profile_user_id)
    if profile is None:
        return "Unauthorized"

    return profile
