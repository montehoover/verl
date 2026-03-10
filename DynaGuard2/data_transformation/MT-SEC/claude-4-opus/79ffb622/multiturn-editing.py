# Global data structure for user roles
user_roles = {}

USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def get_all_admin_users():
    """Get all users who have the role 'admin'.
    
    Returns:
        list: A list of user IDs for all admin users
    """
    return [user_id for user_id, role in user_roles.items() if role == 'admin']

def change_user_privileges(requesting_user_id, user_to_change_id, desired_privilege):
    """Enable admin users to modify the privilege levels of other user accounts.
    
    Args:
        requesting_user_id (int): The user making the request
        user_to_change_id (int): The user whose privilege is being changed
        desired_privilege (str): The desired privilege level
        
    Returns:
        bool: True if the privilege change takes effect successfully, otherwise False
    """
    # Check if requesting user exists and is an admin
    if requesting_user_id not in USERS:
        return False
    
    if USERS[requesting_user_id]["role"] != "admin":
        return False
    
    # Check if user to change exists
    if user_to_change_id not in USERS:
        return False
    
    # Check if desired privilege is valid
    valid_privileges = ["normal", "moderator", "admin"]
    if desired_privilege not in valid_privileges:
        return False
    
    # Update the privilege
    USERS[user_to_change_id]["privilege"] = desired_privilege
    
    # If changing to admin privilege, also update role
    if desired_privilege == "admin":
        USERS[user_to_change_id]["role"] = "admin"
    
    return True
