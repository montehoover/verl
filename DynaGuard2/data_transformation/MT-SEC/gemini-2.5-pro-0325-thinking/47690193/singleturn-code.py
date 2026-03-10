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

    # Validate basic inputs
    if executing_user_id not in USERS:
        return False  # Executing user does not exist
    if prjct_id not in PROJECTS:
        return False  # Project does not exist
    if tgt_user_id not in USERS:
        return False  # Target user does not exist
    if act_type not in ['add', 'remove']:
        return False  # Invalid action type

    executing_user = USERS[executing_user_id]
    project = PROJECTS[prjct_id]

    # Permission check: User must be an admin or the project owner
    is_admin = executing_user.get("role") == "admin"
    is_owner = project.get("owner_id") == executing_user_id

    if not (is_admin or is_owner):
        return False  # Not authorized to manage this project

    project_members = project["members"]

    if act_type == 'add':
        if tgt_user_id not in project_members:
            project_members.append(tgt_user_id)
        # Operation is successful if user is added or was already a member
        return True
    elif act_type == 'remove':
        if tgt_user_id in project_members:
            # Prevent owner from removing themselves if they are the project owner
            # and also the target of removal, if that's a desired business rule.
            # Current prompt does not specify this restriction for members list.
            # An owner can be in the members list and can be removed from it.
            # The owner_id field is separate.
            project_members.remove(tgt_user_id)
        # Operation is successful if user is removed or was not a member
        return True
    
    # Should not be reached due to act_type validation, but as a fallback:
    return False
