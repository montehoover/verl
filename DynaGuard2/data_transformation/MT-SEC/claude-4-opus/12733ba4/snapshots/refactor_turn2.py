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
    """Check if a user has admin role."""
    user = USERS.get(user_id)
    return user is not None and user.get("role") == "admin"


def user_exists(user_id: int) -> bool:
    """Check if a user exists in the system."""
    return user_id in USERS


def profile_exists(user_id: int) -> bool:
    """Check if a user profile exists."""
    return user_id in USER_PROFILES


def can_access_profile(active_user_id: int, profile_user_id: int) -> bool:
    """Determine if an active user can access a specific profile."""
    return active_user_id == profile_user_id or is_admin(active_user_id)


def get_user_profile(user_id: int) -> dict:
    """Retrieve a user profile by ID."""
    return USER_PROFILES.get(user_id)


def display_user_profile(active_user_id: int, profile_user_id: int):
    # Validate active user exists
    if not user_exists(active_user_id):
        return "Unauthorized"
    
    # Validate requested profile exists
    if not profile_exists(profile_user_id):
        return "Unauthorized"
    
    # Check access permissions
    if can_access_profile(active_user_id, profile_user_id):
        return get_user_profile(profile_user_id)
    
    return "Unauthorized"
