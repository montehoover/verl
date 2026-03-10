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
    Allows project owners to add or remove members from their projects,
    while admins can manage any project.

    Args:
        executing_user_id: The ID of the user performing the action.
        prjct_id: The ID of the project.
        act_type: The action to perform ('add' or 'remove').
        tgt_user_id: The ID of the user affected by the change.

    Returns:
        True if the operation was successfully completed, False otherwise.
    """
    # Validate inputs
    if executing_user_id not in USERS or prjct_id not in PROJECTS or tgt_user_id not in USERS:
        return False

    project = PROJECTS[prjct_id]
    executing_user = USERS[executing_user_id]

    # Check permissions
    is_admin = executing_user["role"] == "admin"
    is_owner = project["owner_id"] == executing_user_id

    if not (is_admin or is_owner):
        return False  # User is not admin and not project owner

    # Perform action
    if act_type == "add":
        if tgt_user_id not in project["members"]:
            project["members"].append(tgt_user_id)
        return True
    elif act_type == "remove":
        # Project owner cannot be removed
        if tgt_user_id == project["owner_id"]:
            return False
        if tgt_user_id in project["members"]:
            project["members"].remove(tgt_user_id)
        return True
    else:
        return False # Invalid action type
