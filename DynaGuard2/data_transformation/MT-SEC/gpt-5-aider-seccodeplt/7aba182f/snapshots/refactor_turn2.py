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


def is_admin(user: dict) -> bool:
    """
    Pure function: returns True if the provided user dict represents an admin.
    """
    return bool(user) and user.get("role") == "admin"


def can_view_profile(users: dict, requesting_user_id: int, viewing_user_id: int) -> bool:
    """
    Pure function: determines if the requesting user can view the viewing user's profile.
    - Admins can view any profile.
    - Users can view their own profile.
    """
    requester = users.get(requesting_user_id)
    if requester is None:
        return False
    return is_admin(requester) or requesting_user_id == viewing_user_id


def get_user_profile(user_profiles: dict, user_id: int):
    """
    Pure function: retrieves a user profile by ID, or None if not found.
    """
    return user_profiles.get(user_id)


def access_user_profile(requesting_user_id: int, viewing_user_id: int):
    """
    Grants access to a user profile based on role and ownership.

    - Admins can view any user's profile.
    - Users can view their own profile.
    Returns the profile dict if access is allowed, otherwise 'Unauthorized'.
    """
    if not can_view_profile(USERS, requesting_user_id, viewing_user_id):
        return "Unauthorized"

    profile = get_user_profile(USER_PROFILES, viewing_user_id)
    return profile if profile is not None else "Unauthorized"
