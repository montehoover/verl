# Global data structure for user roles and privileges
USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def get_user_privilege(user_id):
    """Check the current privilege level of a user.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        str: The privilege level of the user
    """
    if user_id in USERS:
        return USERS[user_id]["privilege"]
    return "none"

def generate_user_report():
    """Generate a report of all user IDs and their privilege levels.
    
    Returns:
        list: A list of dictionaries containing user_id and privilege level
    """
    return [{'user_id': user_data["user_id"], 'privilege': user_data["privilege"]} 
            for user_data in USERS.values()]

def modify_user_privileges(current_user_id, target_user_id, new_privilege):
    """Allow only admins to modify the privileges of user accounts.
    
    Args:
        current_user_id (int): The user attempting the change
        target_user_id (int): The user whose privileges are being modified
        new_privilege (str): The new privilege level to assign
        
    Returns:
        bool: True if the privilege was successfully modified, else False
    """
    if current_user_id not in USERS or target_user_id not in USERS:
        return False
    
    if USERS[current_user_id]["privilege"] != "admin":
        return False
    
    USERS[target_user_id]["privilege"] = new_privilege
    return True
