# Global data structure to track user roles
user_roles = {}

def is_user_admin(user_id):
    """Check if a given user is an admin.
    
    Args:
        user_id (int): The ID of the user to check.
        
    Returns:
        bool: True if the user is an admin, False otherwise.
    """
    return user_roles.get(user_id) == 'admin'
