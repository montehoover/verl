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

def fetch_profile_data(requester_id: int, profile_id: int):
    """
    Allows users to view their own profile and administrators to view any profile.

    Args:
        requester_id: The ID of the user making the request.
        profile_id: The ID of the profile being requested.

    Returns:
        The profile details if access is granted, otherwise 'Unauthorized'.
    """
    # Check if the requester and profile exist
    if requester_id not in MEMBER_DATA or profile_id not in MEMBER_PROFILES:
        return "Unauthorized"

    requester_info = MEMBER_DATA[requester_id]
    
    # Grant access if the requester is an admin or is requesting their own profile
    if requester_info["role"] == "admin" or requester_id == profile_id:
        return MEMBER_PROFILES[profile_id]
    else:
        return "Unauthorized"
