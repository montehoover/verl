USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def change_user_privileges(requesting_user_id: int, user_to_change_id: int, desired_privilege: str) -> bool:
    """
    Enables admin users to modify the privilege levels of other user accounts.
    
    Args:
        requesting_user_id: The ID of the user making the request
        user_to_change_id: The ID of the user whose privilege is being changed
        desired_privilege: The desired privilege level for the target user
        
    Returns:
        True if the privilege change takes effect successfully, otherwise False
    """
    # Check if requesting user exists and is an admin
    if requesting_user_id not in USERS:
        return False
    
    requesting_user = USERS[requesting_user_id]
    if requesting_user.get("privilege") != "admin":
        return False
    
    # Check if the user to change exists
    if user_to_change_id not in USERS:
        return False
    
    # Check if the desired privilege is valid
    valid_privileges = ["normal", "moderator", "admin"]
    if desired_privilege not in valid_privileges:
        return False
    
    # Update the privilege
    USERS[user_to_change_id]["privilege"] = desired_privilege
    return True
