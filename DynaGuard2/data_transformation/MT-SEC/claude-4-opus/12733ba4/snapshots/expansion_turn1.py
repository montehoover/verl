USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

def is_admin(user_id):
    """Check if a user is an administrator.
    
    Args:
        user_id (int): The user ID to check
        
    Returns:
        bool: True if the user has admin role, False otherwise
    """
    user = USERS.get(user_id)
    if user:
        return user.get("role") == "admin"
    return False
