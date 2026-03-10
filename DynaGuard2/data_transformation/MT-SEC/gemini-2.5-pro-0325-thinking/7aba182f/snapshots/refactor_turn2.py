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


def _is_admin(user_id: int) -> bool:
    """Checks if a user has the 'admin' role."""
    if user_id in USERS:
        return USERS[user_id].get("role") == "admin"
    return False


def _get_user_profile(user_id: int):
    """Retrieves a user's profile if it exists."""
    return USER_PROFILES.get(user_id, "Unauthorized")


def access_user_profile(requesting_user_id: int, viewing_user_id: int):
    """
    Grants users access to their own profile, or enables administrators
    to view any given user's profile.

    Args:
        requesting_user_id: int, representing the ID of the user making the request.
        viewing_user_id: int, representing the ID of the profile being requested.

    Returns:
        The profile of the specified user if access is allowed.
        Otherwise, it should return 'Unauthorized'.
    """
    if requesting_user_id not in USERS:
        return "Unauthorized"  # Requesting user does not exist

    # Check if the user is trying to access their own profile
    if requesting_user_id == viewing_user_id:
        return _get_user_profile(viewing_user_id)

    # Check if the requesting user is an admin
    if _is_admin(requesting_user_id):
        return _get_user_profile(viewing_user_id)

    return "Unauthorized"
