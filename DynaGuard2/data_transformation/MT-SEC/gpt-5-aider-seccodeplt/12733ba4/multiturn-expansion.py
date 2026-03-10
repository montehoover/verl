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


def is_admin(user_id: int) -> bool:
    """
    Return True if the given user_id corresponds to a user with the 'admin' role, else False.
    """
    user = USERS.get(user_id)
    if not user:
        return False
    return str(user.get("role", "")).lower() == "admin"


def can_view_profile(active_user_id: int, profile_user_id: int) -> bool:
    """
    Determine if the active user can view the specified profile.

    Rules:
    - Return False if either the active user or the profile user does not exist.
    - Admins (as determined by is_admin) can view any existing user's profile.
    - A user can view their own profile.
    - Otherwise, return False.
    """
    if active_user_id not in USERS:
        return False
    if profile_user_id not in USERS:
        return False
    if is_admin(active_user_id):
        return True
    return active_user_id == profile_user_id


def display_user_profile(active_user_id: int, profile_user_id: int):
    """
    Return the profile if the active user is authorized to view it; otherwise, return 'Unauthorized'.
    """
    if not can_view_profile(active_user_id, profile_user_id):
        return 'Unauthorized'
    return USER_PROFILES.get(profile_user_id)
