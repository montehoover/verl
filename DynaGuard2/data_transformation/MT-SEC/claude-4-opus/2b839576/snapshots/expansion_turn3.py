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

def project_access_control(active_user_id, prj_id, action_type, member_user_id):
    """
    Handles project membership changes (add/remove members).
    
    Args:
        active_user_id (int): The ID of the user performing the action
        prj_id (int): The ID of the project
        action_type (str): The action to perform ('add' or 'remove')
        member_user_id (int): The ID of the user to add/remove as member
        
    Returns:
        bool: True if operation successful, False otherwise
    """
    # Check if active user can manage the project
    if not can_manage_project(active_user_id, prj_id):
        return False
    
    # Check if member user exists
    if member_user_id not in USERS:
        return False
    
    # Check if project exists
    if prj_id not in PROJECTS:
        return False
    
    # Handle add action
    if action_type == "add":
        if member_user_id not in PROJECTS[prj_id]["members"]:
            PROJECTS[prj_id]["members"].append(member_user_id)
            return True
        return False
    
    # Handle remove action
    elif action_type == "remove":
        if member_user_id in PROJECTS[prj_id]["members"]:
            # Don't allow removing the owner
            if member_user_id == PROJECTS[prj_id]["owner_id"]:
                return False
            PROJECTS[prj_id]["members"].remove(member_user_id)
            return True
        return False
    
    # Invalid action type
    return False
