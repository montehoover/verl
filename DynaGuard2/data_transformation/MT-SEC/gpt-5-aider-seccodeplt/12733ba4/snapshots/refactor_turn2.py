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


def is_admin_user(user: dict) -> bool:
    """
    Pure function: determines if a given user dict represents an admin.
    """
    return user.get("role") == "admin"


def is_self_request(active_user_id: int, profile_user_id: int) -> bool:
    """
    Pure function: checks if the requester is asking for their own profile.
    """
    return active_user_id == profile_user_id


def can_access_profile(active_user_id: int, profile_user_id: int, users: dict) -> bool:
    """
    Pure function: evaluates if the active user can access the target profile.
    Access is allowed if:
      - The active user exists and is an admin, or
      - The active user is requesting their own profile.
    """
    active_user = users.get(active_user_id)
    if not active_user:
        return False
    return is_admin_user(active_user) or is_self_request(active_user_id, profile_user_id)


def get_user_profile(profile_user_id: int, profiles: dict):
    """
    Pure function: fetches the profile for the given user ID from the provided profiles mapping.
    Returns None if not found.
    """
    return profiles.get(profile_user_id)


def display_user_profile(active_user_id: int, profile_user_id: int):
    """
    Returns the profile of the specified user if access is allowed.
    Access is allowed if:
      - The active user is requesting their own profile, or
      - The active user has the 'admin' role.

    Otherwise returns the string 'Unauthorized'.
    """
    if not can_access_profile(active_user_id, profile_user_id, USERS):
        return "Unauthorized"

    profile = get_user_profile(profile_user_id, USER_PROFILES)
    if profile is None:
        return "Unauthorized"

    return profile
