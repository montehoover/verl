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


def project_access_control(active_user_id: int, prj_id: int, action_type: str, member_user_id: int) -> bool:
    """
    Manages project membership with role-based access control.
    
    Args:
        active_user_id: The user executing the action
        prj_id: The ID of the project being updated
        action_type: Either 'add' or 'remove'
        member_user_id: The user_id of the member being targeted
    
    Returns:
        True if the operation was successfully completed, False otherwise
    """
    # Check if the project exists
    if prj_id not in PROJECTS:
        return False
    
    # Check if both users exist
    if active_user_id not in USERS or member_user_id not in USERS:
        return False
    
    # Check if action_type is valid
    if action_type not in ['add', 'remove']:
        return False
    
    # Get user and project information
    active_user = USERS[active_user_id]
    project = PROJECTS[prj_id]
    
    # Check if user has permission (admin or project owner)
    is_admin = active_user["role"] == "admin"
    is_owner = project["owner_id"] == active_user_id
    
    if not (is_admin or is_owner):
        return False
    
    # Perform the action
    if action_type == 'add':
        if member_user_id not in project["members"]:
            project["members"].append(member_user_id)
        return True
    elif action_type == 'remove':
        if member_user_id in project["members"]:
            project["members"].remove(member_user_id)
        return True
    
    return False
