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

def update_project_members(acting_user_id: int, project_identifier: int, modification: str, user_to_update_id: int) -> bool:
    """
    Allows project owners to add or remove team members from their projects.
    Admins can manage any project without restrictions.
    
    Args:
        acting_user_id: The ID of the user performing the task
        project_identifier: The ID of the project to be modified
        modification: Either 'add' or 'remove' indicating the operation
        user_to_update_id: The ID of the user to be added/removed
    
    Returns:
        True if the modification is successful, otherwise False
    """
    # Check if the acting user exists
    if acting_user_id not in USERS:
        return False
    
    # Check if the project exists
    if project_identifier not in PROJECTS:
        return False
    
    # Check if the user to update exists
    if user_to_update_id not in USERS:
        return False
    
    # Check if modification is valid
    if modification not in ['add', 'remove']:
        return False
    
    # Get user role and project info
    acting_user = USERS[acting_user_id]
    project = PROJECTS[project_identifier]
    
    # Check permissions: user must be admin or project owner
    is_admin = acting_user["role"] == "admin"
    is_owner = project["owner_id"] == acting_user_id
    
    if not (is_admin or is_owner):
        return False
    
    # Perform the modification
    if modification == 'add':
        if user_to_update_id not in project["members"]:
            project["members"].append(user_to_update_id)
            return True
        else:
            # User is already a member
            return False
    else:  # modification == 'remove'
        if user_to_update_id in project["members"]:
            project["members"].remove(user_to_update_id)
            return True
        else:
            # User is not a member
            return False
