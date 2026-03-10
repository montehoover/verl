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

    requesting_user = USERS[requesting_user_id]

    # Check if the user is trying to access their own profile
    if requesting_user_id == viewing_user_id:
        if viewing_user_id in USER_PROFILES:
            return USER_PROFILES[viewing_user_id]
        else:
            return "Unauthorized" # Profile to view does not exist

    # Check if the requesting user is an admin
    if requesting_user.get("role") == "admin":
        if viewing_user_id in USER_PROFILES:
            return USER_PROFILES[viewing_user_id]
        else:
            return "Unauthorized" # Profile to view does not exist

    return "Unauthorized"
