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
    """Check if a user is an administrator.
    
    Args:
        user_id (int): The user ID to check
        
    Returns:
        bool: True if the user has admin role, False otherwise
    """
    user = USERS.get(user_id)
    if user:
        return user.get("role") == "admin"
    return False

def can_view_profile(active_user_id, profile_user_id):
    """Determine if a user can view another user's profile.
    
    Args:
        active_user_id (int): The ID of the user trying to view a profile
        profile_user_id (int): The ID of the profile being viewed
        
    Returns:
        bool: True if the active user can view the profile, False otherwise
    """
    # Users can always view their own profile
    if active_user_id == profile_user_id:
        return True
    
    # Admins can view any profile
    if is_admin(active_user_id):
        return True
    
    # Regular users cannot view other users' profiles
    return False

def display_user_profile(active_user_id, profile_user_id):
    """Display a user profile if the active user has permission.
    
    Args:
        active_user_id (int): The ID of the user trying to view a profile
        profile_user_id (int): The ID of the profile to display
        
    Returns:
        dict or str: The user profile if authorized, 'Unauthorized' otherwise
    """
    if can_view_profile(active_user_id, profile_user_id):
        return USER_PROFILES.get(profile_user_id)
    return 'Unauthorized'
