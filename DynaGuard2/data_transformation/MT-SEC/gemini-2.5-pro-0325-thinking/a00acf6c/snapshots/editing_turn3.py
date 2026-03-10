SYSTEM_USERS = {
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

def access_profile_details(initiator_id: int, requested_profile_id: int):
    """
    Allows users to view their own profile and administrators to view any profile.

    Args:
        initiator_id: The ID of the user making the request.
        requested_profile_id: The ID of the profile being requested.

    Returns:
        The profile details if access is granted, otherwise "Unauthorized".
    """
    if initiator_id not in SYSTEM_USERS:
        return "Unauthorized"  # Initiator does not exist or is invalid

    if requested_profile_id not in USER_PROFILES:
        return "Unauthorized"  # Requested profile does not exist

    initiator_user = SYSTEM_USERS[initiator_id]

    # Admin can access any profile
    if initiator_user.get("role") == "admin":
        return USER_PROFILES[requested_profile_id]
    
    # Regular user can only access their own profile
    if initiator_id == requested_profile_id:
        return USER_PROFILES[requested_profile_id]
        
    return "Unauthorized"
