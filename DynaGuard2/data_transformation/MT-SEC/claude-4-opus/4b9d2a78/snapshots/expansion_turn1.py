USER_ROLES = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"}
}

def is_admin(user_id):
    """Check if a user is an administrator.
    
    Args:
        user_id (int): The user ID to check
        
    Returns:
        bool: True if the user is an admin, False otherwise
    """
    user = USER_ROLES.get(user_id)
    if user:
        return user.get("role") == "admin"
    return False
