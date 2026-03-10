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


def manage_team_access(executor_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Manage team access for projects. Allows project owners to add/remove members from their own projects,
    and administrators to modify any project.
    
    Args:
        executor_user_id: The user_id of the individual performing the action
        proj_id: The project being changed
        operation: Either 'add' or 'remove'
        target_member_id: The user_id of the individual being added or removed
        
    Returns:
        True if the operation is successful, otherwise False
    """
    # Check if the executor user exists
    if executor_user_id not in USERS:
        return False
    
    # Check if the project exists
    if proj_id not in PROJECTS:
        return False
    
    # Check if the target member exists
    if target_member_id not in USERS:
        return False
    
    # Check if operation is valid
    if operation not in ['add', 'remove']:
        return False
    
    # Get executor user info
    executor_user = USERS[executor_user_id]
    project = PROJECTS[proj_id]
    
    # Check permissions: admin can modify any project, owner can modify their own project
    is_admin = executor_user["role"] == "admin"
    is_owner = project["owner_id"] == executor_user_id
    
    if not (is_admin or is_owner):
        return False
    
    # Perform the operation
    if operation == 'add':
        if target_member_id not in project["members"]:
            project["members"].append(target_member_id)
            return True
        else:
            # Member already exists in the project
            return False
    elif operation == 'remove':
        if target_member_id in project["members"]:
            project["members"].remove(target_member_id)
            return True
        else:
            # Member not found in the project
            return False
    
    return False
