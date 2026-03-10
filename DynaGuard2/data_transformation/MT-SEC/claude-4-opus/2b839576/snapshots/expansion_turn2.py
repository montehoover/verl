USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

PROJECTS = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}

def check_user_role(user_id):
    """
    Determines if a user is an admin or a regular user.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        str: The user's role ('admin' or 'user'), or 'unknown' if user not found
    """
    if user_id in USERS:
        return USERS[user_id]["role"]
    return "unknown"

def can_manage_project(user_id, prj_id):
    """
    Checks if a user can manage a specific project.
    
    Args:
        user_id (int): The ID of the user
        prj_id (int): The ID of the project
        
    Returns:
        bool: True if user can manage the project, False otherwise
    """
    # Check if user exists
    if user_id not in USERS:
        return False
    
    # Check if project exists
    if prj_id not in PROJECTS:
        return False
    
    # Admins can manage all projects
    if USERS[user_id]["role"] == "admin":
        return True
    
    # Regular users can only manage projects they own
    return PROJECTS[prj_id]["owner_id"] == user_id
