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
    Grants users access to their own profile, or enables administrators to view any given user's profile.

    Args:
        active_user_id: int, representing the ID of the user making the request.
        profile_user_id: int, representing the ID of the profile being requested.

    Returns:
        The profile of the specified user if access is allowed.
        Otherwise, it should return 'Unauthorized'.
    """

    # Check if the user whose profile is being requested exists
    if profile_user_id not in USER_PROFILES:
        return 'Unauthorized'

    # Check if the active user (the one making the request) exists
    active_user = USERS.get(active_user_id)
    if not active_user:
        return 'Unauthorized'

    # Allow access if the active user is requesting their own profile
    if active_user_id == profile_user_id:
        return USER_PROFILES[profile_user_id]

    # Allow access if the active user is an admin
    if active_user.get("role") == "admin":
        return USER_PROFILES[profile_user_id]

    # If none of the above conditions are met, deny access
    return 'Unauthorized'
