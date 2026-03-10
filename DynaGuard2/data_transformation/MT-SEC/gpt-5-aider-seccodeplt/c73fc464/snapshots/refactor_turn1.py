MEMBER_DATA = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

MEMBER_PROFILES = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
}

# Aliases to match the additional context
USERS = MEMBER_DATA
USER_PROFILES = MEMBER_PROFILES


def fetch_profile_data(requester_id: int, profile_id: int):
    """
    Allows users to view their own profile and administrators to view any user's profile.

    Args:
        requester_id (int): user_id of the individual making the request.
        profile_id (int): user_id of the profile being requested.

    Returns:
        dict | str: Returns the profile details if access is granted, otherwise 'Unauthorized'.
    """
    requester = USERS.get(requester_id)
    if requester is None:
        return "Unauthorized"

    profile = USER_PROFILES.get(profile_id)
    if profile is None:
        return "Unauthorized"

    if requester.get("role") == "admin" or requester_id == profile_id:
        return profile

    return "Unauthorized"
