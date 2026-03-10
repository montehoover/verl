# Global data structure to track user roles
user_roles = {}

def can_access_content(user_id):
    """Check if a user can access content based on their role.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        bool: True if the user is an admin, False otherwise
    """
    return user_roles.get(user_id) == 'admin'
