# Global data structure for user roles
user_roles = {}

def get_all_admin_users():
    """Get all users who have the role 'admin'.
    
    Returns:
        list: A list of user IDs for all admin users
    """
    return [user_id for user_id, role in user_roles.items() if role == 'admin']
