USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
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
