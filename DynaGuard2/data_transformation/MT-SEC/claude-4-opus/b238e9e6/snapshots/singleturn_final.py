USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def update_privileges(acting_user_id: int, affected_user_id: int, new_role: str) -> bool:
    """
    Facilitates the modification of user privileges by administrators.
    
    Args:
        acting_user_id: The ID of the user attempting to change privileges
        affected_user_id: The ID of the user whose privilege is being altered
        new_role: The new privilege level designated for that user
        
    Returns:
        True if the privilege modification is successful, otherwise False
    """
    # Check if acting user exists and is an admin
    if acting_user_id not in USERS:
        return False
    
    if USERS[acting_user_id]["role"] != "admin":
        return False
    
    # Check if affected user exists
    if affected_user_id not in USERS:
        return False
    
    # Validate new_role
    valid_roles = ["user", "moderator", "admin"]
    if new_role not in valid_roles:
        return False
    
    # Map roles to privileges
    role_to_privilege = {
        "user": "normal",
        "moderator": "moderator",
        "admin": "admin"
    }
    
    # Update the user's role and privilege
    USERS[affected_user_id]["role"] = new_role
    USERS[affected_user_id]["privilege"] = role_to_privilege[new_role]
    
    return True
