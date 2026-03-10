# Global data structure for user profiles
user_profiles = {}

def view_own_profile(user_id):
    """
    Allows users to view their own profile information.
    
    Args:
        user_id (int): The ID of the user
        
    Returns:
        dict or str: User's profile details if exists, otherwise 'User not found'
    """
    if user_id in user_profiles:
        return user_profiles[user_id]
    else:
        return 'User not found'
