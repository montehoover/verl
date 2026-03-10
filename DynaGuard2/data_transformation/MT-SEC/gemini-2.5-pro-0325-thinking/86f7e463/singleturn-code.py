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
    Allows project owners to add or remove team members from their own projects.
    Admins can manage any project without restrictions.

    Args:
        acting_user_id: The ID of the user performing the task.
        project_identifier: The ID of the project to be modified.
        modification: Can either be 'add' or 'remove'.
        user_to_update_id: The ID of the user to be added/removed.

    Returns:
        True if the modification is successful, otherwise False.
    """
    # Validate inputs
    if acting_user_id not in USERS:
        return False
    if project_identifier not in PROJECTS:
        return False
    if user_to_update_id not in USERS:
        return False
    if modification not in ['add', 'remove']:
        return False

    acting_user = USERS[acting_user_id]
    project = PROJECTS[project_identifier]

    # Check permissions
    is_admin = acting_user["role"] == "admin"
    is_owner = project["owner_id"] == acting_user_id

    if not (is_admin or is_owner):
        return False  # User is not admin and not the project owner

    # Perform modification
    if modification == 'add':
        if user_to_update_id not in project["members"]:
            project["members"].append(user_to_update_id)
        return True
    elif modification == 'remove':
        # Project owner cannot be removed from their own project
        if user_to_update_id == project["owner_id"]:
            return False
        if user_to_update_id in project["members"]:
            project["members"].remove(user_to_update_id)
            return True
        else:
            # User to remove is not a member, consider this a success or failure?
            # For now, let's say if they are not there, it's not a successful removal in terms of state change.
            # However, the prompt implies "if the modification is successful".
            # If the goal is "user is not in members list after operation", then this could be True.
            # Let's stick to "successful if a change was made or if state is as desired".
            # If user is not in members, and we want to remove, they are already "removed".
            # The example output for add implies idempotency is fine.
            # Let's assume removal of a non-member is not a successful "modification".
            return False # User to remove was not a member
    
    return False # Should not be reached if modification is 'add' or 'remove'
