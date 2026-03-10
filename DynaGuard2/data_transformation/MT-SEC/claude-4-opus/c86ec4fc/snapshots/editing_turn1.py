# Global data structure to track user roles
user_roles = {}

def check_user_role(user_id):
    """Determines if a user is an admin or a regular user.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        str: 'admin' if the user is an admin, 'user' if they are a regular user
    """
    return user_roles.get(user_id, 'user')
