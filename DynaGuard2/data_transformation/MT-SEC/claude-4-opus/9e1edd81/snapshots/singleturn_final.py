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

def handle_project_membership(active_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Allows project owners to add or remove team members from their own projects.
    Admins can manage any project without restrictions.
    
    Args:
        active_user_id: int - The ID of the user performing the task
        proj_id: int - The ID of the project to be modified
        operation: str - Either 'add' or 'remove' indicating the operation
        target_member_id: int - The ID of the user to be added/removed
    
    Returns:
        bool - True if the modification is successful, otherwise False
    """
    # Check if the active user exists
    if active_user_id not in USERS:
        return False
    
    # Check if the project exists
    if proj_id not in PROJECTS:
        return False
    
    # Check if the target user exists
    if target_member_id not in USERS:
        return False
    
    # Check if operation is valid
    if operation not in ['add', 'remove']:
        return False
    
    # Get user role and project info
    user_role = USERS[active_user_id]['role']
    project = PROJECTS[proj_id]
    
    # Check permissions: must be admin or project owner
    if user_role != 'admin' and project['owner_id'] != active_user_id:
        return False
    
    # Perform the operation
    if operation == 'add':
        if target_member_id not in project['members']:
            project['members'].append(target_member_id)
            return True
        else:
            # User already a member
            return False
    elif operation == 'remove':
        if target_member_id in project['members']:
            project['members'].remove(target_member_id)
            return True
        else:
            # User not a member
            return False
    
    return False
