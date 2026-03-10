# Global data structure for user profiles
user_profiles = {}

def view_own_profile(user_id):
    """
    Allows users to view their own profile details.
    
    Args:
        user_id (int): The ID of the user
        
    Returns:
        dict or str: Profile details if user exists, otherwise 'Profile not found'
    """
    if user_id in user_profiles:
        return user_profiles[user_id]
    else:
        return 'Profile not found'
