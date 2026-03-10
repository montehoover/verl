# Global data structure for user roles and privileges
user_privileges = {}

def get_user_privilege(user_id):
    """Check the current privilege level of a user.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        str: The privilege level of the user
    """
    return user_privileges.get(user_id, "none")
