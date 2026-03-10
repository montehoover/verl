USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def is_admin(user_id):
    """Check if a user has admin privileges.
    
    Args:
        user_id (int): The ID of the user to check.
        
    Returns:
        bool: True if the user is an admin, False otherwise.
    """
    user = USERS.get(user_id)
    if user:
        return user.get("privilege") == "admin"
    return False

def can_change_privilege(requesting_user_id, user_to_change_id, desired_privilege):
    """Check if a privilege change is allowed.
    
    Args:
        requesting_user_id (int): The ID of the user requesting the change.
        user_to_change_id (int): The ID of the user whose privilege is to be changed.
        desired_privilege (str): The new privilege to assign.
        
    Returns:
        bool: True if the privilege change is allowed, False otherwise.
    """
    # Check if the requesting user is an admin
    if not is_admin(requesting_user_id):
        return False
    
    # Check if the user to change exists
    if user_to_change_id not in USERS:
        return False
    
    # Admin can change any user's privilege
    return True
