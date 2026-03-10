# Global data structures
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

# Global data structure to store user profiles
user_profiles = {}

def can_view_own_profile(user_id):
    """
    Check if a user can view their own profile.
    
    Args:
        user_id (int): The ID of the user
        
    Returns:
        dict: The user's profile if authorized
        str: 'Unauthorized' if the user cannot view their profile
    """
    if user_id in user_profiles:
        return user_profiles[user_id]
    return 'Unauthorized'

def view_user_profile(requesting_user_id, target_user_id):
    """
    Check if a requesting user can view a target user's profile.
    
    Args:
        requesting_user_id (int): The ID of the user making the request
        target_user_id (int): The ID of the user whose profile is being requested
        
    Returns:
        dict: The target user's profile if authorized
        str: 'Unauthorized' if the requesting user cannot view the profile
    """
    # Allow users to view their own profile
    if requesting_user_id == target_user_id:
        return can_view_own_profile(requesting_user_id)
    
    # Check if target user exists
    if target_user_id in user_profiles:
        # Add logic here for additional authorization rules
        # For now, only allow viewing own profile
        return 'Unauthorized'
    
    return 'Unauthorized'

def access_user_profile(requesting_user_id, viewing_user_id):
    """
    Allow users to access their own profile, while administrators can view any user's profile.
    
    Args:
        requesting_user_id (int): The ID of the user making the request
        viewing_user_id (int): The ID of the profile being requested
        
    Returns:
        dict: The profile of the specified user if access is allowed
        str: 'Unauthorized' if access is not allowed
    """
    # Check if requesting user exists
    if requesting_user_id not in USERS:
        return 'Unauthorized'
    
    # Check if the profile being requested exists
    if viewing_user_id not in USER_PROFILES:
        return 'Unauthorized'
    
    # Allow users to view their own profile
    if requesting_user_id == viewing_user_id:
        return USER_PROFILES[viewing_user_id]
    
    # Check if requesting user is an admin
    if USERS[requesting_user_id]["role"] == "admin":
        return USER_PROFILES[viewing_user_id]
    
    # Default to unauthorized
    return 'Unauthorized'
