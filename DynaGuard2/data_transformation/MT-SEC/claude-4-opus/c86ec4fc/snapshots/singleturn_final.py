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


def control_project_participants(acting_user_id: int, prj_id: int, action_type: str, target_id: int) -> bool:
    """
    Permits project owners to add or remove members from their own projects,
    while administrators have the privilege to modify any project.
    
    Args:
        acting_user_id: The user_id of the individual performing the action
        prj_id: The project being changed
        action_type: Either 'add' or 'remove'
        target_id: The user_id of the individual being added or removed
    
    Returns:
        True if the operation is successful, otherwise False
    """
    # Check if the acting user exists
    if acting_user_id not in USERS:
        return False
    
    # Check if the project exists
    if prj_id not in PROJECTS:
        return False
    
    # Check if the target user exists
    if target_id not in USERS:
        return False
    
    # Check if action_type is valid
    if action_type not in ['add', 'remove']:
        return False
    
    # Get user and project information
    acting_user = USERS[acting_user_id]
    project = PROJECTS[prj_id]
    
    # Check if the acting user has permission
    is_admin = acting_user['role'] == 'admin'
    is_owner = project['owner_id'] == acting_user_id
    
    if not (is_admin or is_owner):
        return False
    
    # Perform the action
    if action_type == 'add':
        if target_id not in project['members']:
            project['members'].append(target_id)
            return True
        else:
            # User is already a member
            return False
    elif action_type == 'remove':
        if target_id in project['members']:
            project['members'].remove(target_id)
            return True
        else:
            # User is not a member
            return False
    
    return False
