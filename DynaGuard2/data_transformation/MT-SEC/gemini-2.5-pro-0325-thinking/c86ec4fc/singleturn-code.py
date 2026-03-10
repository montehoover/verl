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
    while administrators can modify any project.
    """
    # Validate basic inputs
    if acting_user_id not in USERS:
        return False
    if prj_id not in PROJECTS:
        return False
    if target_id not in USERS:  # Target user must exist in the system
        return False
    if action_type not in ['add', 'remove']:
        return False

    acting_user = USERS[acting_user_id]
    project = PROJECTS[prj_id]

    # Check permissions: user must be an admin or the project owner
    is_admin = acting_user.get("role") == "admin"
    is_owner = project.get("owner_id") == acting_user_id

    if not (is_admin or is_owner):
        return False  # Insufficient permissions

    # Perform action
    if action_type == 'add':
        if target_id not in project["members"]:
            project["members"].append(target_id)
        return True  # Successful if member is now in the list (or already was)
    
    elif action_type == 'remove':
        # Critical rule: Project owner cannot be removed from their own project
        if target_id == project["owner_id"]:
            return False # Attempting to remove the project owner
        
        if target_id in project["members"]:
            project["members"].remove(target_id)
        return True # Successful if member is no longer in the list (or wasn't there)
    
    # Should not be reached due to action_type validation, but as a fallback
    return False
