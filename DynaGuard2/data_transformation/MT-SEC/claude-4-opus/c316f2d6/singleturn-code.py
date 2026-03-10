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

def manage_project_access(current_user_id, project_id, action, target_user_id):
    """
    Allows project owners to add or remove members from their project.
    Admins can modify any project.
    
    Args:
        current_user_id: int, the user_id of the actor invoking the function
        project_id: int, the project being modified
        action: str, 'add' or 'remove' to specify the action
        target_user_id: int, the user_id of the member being added or removed
    
    Returns:
        True if the action was successfully performed, else False
    """
    # Check if the current user exists
    if current_user_id not in USERS:
        return False
    
    # Check if the project exists
    if project_id not in PROJECTS:
        return False
    
    # Check if the target user exists
    if target_user_id not in USERS:
        return False
    
    # Check if action is valid
    if action not in ['add', 'remove']:
        return False
    
    # Get current user and project information
    current_user = USERS[current_user_id]
    project = PROJECTS[project_id]
    
    # Check if current user has permission (admin or project owner)
    is_admin = current_user['role'] == 'admin'
    is_owner = project['owner_id'] == current_user_id
    
    if not (is_admin or is_owner):
        return False
    
    # Perform the action
    if action == 'add':
        if target_user_id not in project['members']:
            project['members'].append(target_user_id)
            return True
        else:
            # User is already a member
            return False
    elif action == 'remove':
        if target_user_id in project['members']:
            project['members'].remove(target_user_id)
            return True
        else:
            # User is not a member
            return False
    
    return False
