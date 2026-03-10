# Global data structure to track user roles
user_roles = {}

def get_user_role(user_id):
    """
    Determines whether a user is an admin or a regular user.
    
    Args:
        user_id (int): The ID of the user
        
    Returns:
        str: 'admin' if the user is an admin, 'user' otherwise
    """
    return user_roles.get(user_id, 'user')
