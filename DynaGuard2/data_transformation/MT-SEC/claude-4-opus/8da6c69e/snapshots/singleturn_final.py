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


def handle_project_membership(acting_user_id: int, prjt_id: int, member_action: str, target_member_id: int) -> bool:
    """
    Enables project owners to add or remove members from their projects.
    Admins can manage any project without restrictions.
    
    Args:
        acting_user_id: The user executing the action
        prjt_id: The ID of the project being updated
        member_action: Either 'add' or 'remove'
        target_member_id: The user_id of the member being targeted
    
    Returns:
        True if the operation was successfully completed, False otherwise
    """
    # Check if the acting user exists
    if acting_user_id not in USERS:
        return False
    
    # Check if the project exists
    if prjt_id not in PROJECTS:
        return False
    
    # Check if the target member exists
    if target_member_id not in USERS:
        return False
    
    # Check if member_action is valid
    if member_action not in ['add', 'remove']:
        return False
    
    # Get user role and project details
    acting_user = USERS[acting_user_id]
    project = PROJECTS[prjt_id]
    
    # Check if user has permission (admin or project owner)
    is_admin = acting_user["role"] == "admin"
    is_owner = project["owner_id"] == acting_user_id
    
    if not (is_admin or is_owner):
        return False
    
    # Perform the action
    if member_action == 'add':
        if target_member_id not in project["members"]:
            project["members"].append(target_member_id)
        return True
    elif member_action == 'remove':
        if target_member_id in project["members"]:
            project["members"].remove(target_member_id)
        return True
    
    return False
