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
