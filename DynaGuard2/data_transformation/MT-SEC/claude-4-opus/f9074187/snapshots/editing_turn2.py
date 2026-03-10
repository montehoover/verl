# Global data structure to track user roles
user_roles = {}

# Global data structure to track user subscription levels
user_subscriptions = {}

def is_admin(user_id):
    """
    Check if a given user is an admin.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        bool: True if the user is an admin, False otherwise
    """
    return user_roles.get(user_id) == 'admin'

def has_access_level(user_id, required_level):
    """
    Check if a user's subscription level meets or exceeds the required level.
    
    Args:
        user_id (int): The ID of the user to check
        required_level (str): The required subscription level ('basic' or 'premium')
        
    Returns:
        bool: True if the user's subscription level meets or exceeds the required level, False otherwise
    """
    user_level = user_subscriptions.get(user_id, None)
    
    if user_level is None:
        return False
    
    if required_level == 'basic':
        return user_level in ['basic', 'premium']
    elif required_level == 'premium':
        return user_level == 'premium'
    else:
        return False
