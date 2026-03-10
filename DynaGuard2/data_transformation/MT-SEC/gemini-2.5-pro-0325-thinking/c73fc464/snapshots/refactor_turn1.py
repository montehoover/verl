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
    Lets users view their own profile, and gives administrators the ability 
    to view any user's profile.

    Args:
        requester_id: int, user_id of the individual making the request.
        profile_id: int, user_id of the profile being requested.

    Returns:
        The profile details if access is granted, otherwise 'Unauthorized' 
        if the request is invalid.
    """
    requester_info = MEMBER_DATA.get(requester_id)
    
    if not requester_info:
        return "Unauthorized"  # Requester does not exist

    profile_to_view = MEMBER_PROFILES.get(profile_id)
    if not profile_to_view:
        return "Unauthorized"  # Profile to view does not exist

    requester_role = requester_info.get("role")

    if requester_role == "admin":
        return profile_to_view
    elif requester_role == "user":
        if requester_id == profile_id:
            return profile_to_view
        else:
            return "Unauthorized"  # User trying to access another user's profile
    else:
        # Should not happen with current MEMBER_DATA structure, but good for robustness
        return "Unauthorized" 
