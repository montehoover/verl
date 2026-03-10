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

def generate_user_report():
    """Generate a report of all user IDs and their privilege levels.
    
    Returns:
        list: A list of dictionaries containing user_id and privilege level
    """
    return [{'user_id': user_id, 'privilege': privilege} 
            for user_id, privilege in user_privileges.items()]
