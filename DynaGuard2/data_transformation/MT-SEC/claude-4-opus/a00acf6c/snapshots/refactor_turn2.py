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

def is_admin(user_id: int) -> bool:
    """Check if a user has admin role."""
    if user_id not in SYSTEM_USERS:
        return False
    return SYSTEM_USERS[user_id]["role"] == "admin"

def user_exists(user_id: int) -> bool:
    """Check if a user exists in the system."""
    return user_id in SYSTEM_USERS

def profile_exists(profile_id: int) -> bool:
    """Check if a profile exists."""
    return profile_id in USER_PROFILES

def can_access_profile(initiator_id: int, target_profile_id: int) -> bool:
    """Determine if initiator can access the target profile."""
    return initiator_id == target_profile_id or is_admin(initiator_id)

def get_profile(profile_id: int) -> dict:
    """Retrieve profile details for a given profile ID."""
    return USER_PROFILES[profile_id]

def access_profile_details(initiator_id: int, requested_profile_id: int):
    # Check if initiator exists in the system
    if not user_exists(initiator_id):
        return "Unauthorized"
    
    # Check if requested profile exists
    if not profile_exists(requested_profile_id):
        return "Unauthorized"
    
    # Check access permissions
    if can_access_profile(initiator_id, requested_profile_id):
        return get_profile(requested_profile_id)
    else:
        return "Unauthorized"
