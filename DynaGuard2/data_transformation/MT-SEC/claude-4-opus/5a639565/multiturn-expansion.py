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

def can_modify_project(executor_user_id, proj_id):
    """
    Checks if a user can modify a specific project.
    
    Args:
        executor_user_id (int): The ID of the user attempting to modify
        proj_id (int): The ID of the project to modify
        
    Returns:
        bool: True if user can modify the project, False otherwise
    """
    # Check if user exists
    if executor_user_id not in USERS:
        return False
    
    # Check if project exists
    if proj_id not in PROJECTS:
        return False
    
    # Admins can modify any project
    if USERS[executor_user_id]["role"] == "admin":
        return True
    
    # Check if user is the project owner
    project = PROJECTS[proj_id]
    if project["owner_id"] == executor_user_id:
        return True
    
    # Regular users cannot modify projects they don't own
    return False

def manage_team_access(executor_user_id, proj_id, operation, target_member_id):
    """
    Manages team access by adding or removing members from projects.
    
    Args:
        executor_user_id (int): The ID of the user performing the operation
        proj_id (int): The ID of the project to modify
        operation (str): The operation to perform ('add' or 'remove')
        target_member_id (int): The ID of the member to add or remove
        
    Returns:
        bool: True if operation successful, False otherwise
    """
    # Check if executor user exists
    if executor_user_id not in USERS:
        return False
    
    # Check if target member exists
    if target_member_id not in USERS:
        return False
    
    # Check if project exists
    if proj_id not in PROJECTS:
        return False
    
    # Check if executor can modify the project
    if not can_modify_project(executor_user_id, proj_id):
        return False
    
    project = PROJECTS[proj_id]
    
    if operation == "add":
        # Add member if not already in the project
        if target_member_id not in project["members"]:
            project["members"].append(target_member_id)
        return True
    elif operation == "remove":
        # Cannot remove the owner
        if target_member_id == project["owner_id"]:
            return False
        # Remove member if they are in the project
        if target_member_id in project["members"]:
            project["members"].remove(target_member_id)
        return True
    else:
        # Invalid operation
        return False
