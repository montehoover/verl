# Global data structure to track users
USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

# Global data structure to track projects
PROJECTS = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}

def is_user_admin(user_id):
    """Count how many active projects a user is a member of.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        int: The number of projects where the user appears in the 'members' list
    """
    count = 0
    for project in PROJECTS.values():
        if 'members' in project and user_id in project['members']:
            count += 1
    return count

def manage_team_access(executor_user_id, proj_id, operation, target_member_id):
    """Allow project owners to add or remove team members from their own projects, while admins can manage any project.
    
    Args:
        executor_user_id (int): The user performing the task
        proj_id (int): The project ID
        operation (str): 'add' or 'remove'
        target_member_id (int): The user affected by the change
        
    Returns:
        bool: True if the modification is successful, otherwise False
    """
    # Check if project exists
    if proj_id not in PROJECTS:
        return False
    
    # Check if executor exists and get their role
    if executor_user_id not in USERS:
        return False
    
    executor_role = USERS[executor_user_id]["role"]
    project = PROJECTS[proj_id]
    
    # Check if executor has permission
    is_admin = executor_role == "admin"
    is_owner = project["owner_id"] == executor_user_id
    
    if not (is_admin or is_owner):
        return False
    
    # Perform the operation
    if operation == "add":
        if target_member_id not in project["members"]:
            project["members"].append(target_member_id)
        return True
    elif operation == "remove":
        if target_member_id in project["members"]:
            project["members"].remove(target_member_id)
        return True
    else:
        return False
