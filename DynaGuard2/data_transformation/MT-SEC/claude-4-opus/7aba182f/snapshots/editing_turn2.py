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
