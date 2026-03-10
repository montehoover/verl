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

def is_admin(user_id):
    """Check if a user has admin role."""
    if user_id not in USERS:
        return False
    return USERS[user_id]["role"] == "admin"

def user_exists(user_id):
    """Check if a user exists in the system."""
    return user_id in USERS

def profile_exists(user_id):
    """Check if a user profile exists."""
    return user_id in USER_PROFILES

def get_user_profile(user_id):
    """Retrieve a user profile by ID."""
    return USER_PROFILES.get(user_id)

def can_access_profile(requesting_user_id, viewing_user_id):
    """Determine if a user can access a specific profile."""
    # User can access their own profile
    if requesting_user_id == viewing_user_id:
        return True
    
    # Admin can access any profile
    return is_admin(requesting_user_id)

def access_user_profile(requesting_user_id, viewing_user_id):
    # Check if requesting user exists
    if not user_exists(requesting_user_id):
        return "Unauthorized"
    
    # Check if the profile being requested exists
    if not profile_exists(viewing_user_id):
        return "Unauthorized"
    
    # Check access permissions
    if can_access_profile(requesting_user_id, viewing_user_id):
        return get_user_profile(viewing_user_id)
    
    return "Unauthorized"
