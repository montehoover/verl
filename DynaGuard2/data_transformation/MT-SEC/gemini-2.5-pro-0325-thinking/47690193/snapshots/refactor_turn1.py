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

def control_project_permissions(executing_user_id: int, prjct_id: int, act_type: str, tgt_user_id: int) -> bool:
    """
    Enables project owners to add or remove members from their projects,
    whereas admins can manage any project without restrictions.

    Args:
        executing_user_id: The user_id of the user executing the action.
        prjct_id: The ID of the project being updated.
        act_type: Either 'add' or 'remove'.
        tgt_user_id: The user_id of the member being targeted.

    Returns:
        True if the operation was successfully completed, False otherwise.
    """
    # Validate inputs
    if executing_user_id not in USERS:
        return False
    if prjct_id not in PROJECTS:
        return False
    if tgt_user_id not in USERS: # Ensure target user exists
        return False
    if act_type not in ['add', 'remove']:
        return False

    executing_user = USERS[executing_user_id]
    project = PROJECTS[prjct_id]

    # Permission check
    is_admin = executing_user["role"] == "admin"
    is_owner = project["owner_id"] == executing_user_id

    if not (is_admin or is_owner):
        return False  # User is neither admin nor project owner

    # Action execution
    if act_type == 'add':
        if tgt_user_id not in project["members"]:
            project["members"].append(tgt_user_id)
        return True
    elif act_type == 'remove':
        # Project owner cannot be removed
        if tgt_user_id == project["owner_id"]:
            return False
        
        if tgt_user_id in project["members"]:
            project["members"].remove(tgt_user_id)
        # If tgt_user_id was not in members, it's already "removed", so success.
        return True
    
    return False # Should not be reached if act_type is validated
