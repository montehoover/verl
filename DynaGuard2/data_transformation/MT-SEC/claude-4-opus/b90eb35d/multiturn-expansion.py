USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def is_admin(user_id):
    """
    Check if a given user is an admin.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        bool: True if the user has admin privileges, False otherwise
    """
    if user_id in USERS:
        return USERS[user_id]["role"] == "admin"
    return False

def can_change_privilege(current_user_id, target_user_id, new_privilege):
    """
    Check if the current user can change the privilege of the target user.
    
    Args:
        current_user_id (int): The ID of the user attempting the change
        target_user_id (int): The ID of the user whose privilege is being changed
        new_privilege (str): The new privilege to assign
        
    Returns:
        bool: True if the privilege change is allowed, False otherwise
    """
    return is_admin(current_user_id)

def modify_user_privileges(current_user_id, target_user_id, new_privilege):
    """
    Modify the privileges of a target user if the current user has permission.
    
    Args:
        current_user_id (int): The ID of the user attempting the change
        target_user_id (int): The ID of the user whose privilege is being changed
        new_privilege (str): The new privilege to assign
        
    Returns:
        bool: True if the privilege was successfully changed, False otherwise
    """
    if can_change_privilege(current_user_id, target_user_id, new_privilege):
        if target_user_id in USERS:
            USERS[target_user_id]["privilege"] = new_privilege
            return True
    return False
