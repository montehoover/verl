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
        acting_user_id: The ID of the user performing the task.
        project_identifier: The ID of the project to be modified.
        modification: 'add' or 'remove'.
        user_to_update_id: The ID of the user to be added/removed.

    Returns:
        True if the modification is successful, otherwise False.
    """
    # Validate inputs
    if acting_user_id not in USERS:
        return False  # Acting user does not exist
    if project_identifier not in PROJECTS:
        return False  # Project does not exist
    if user_to_update_id not in USERS:
        return False # User to update does not exist
    if modification not in ['add', 'remove']:
        return False # Invalid modification type

    acting_user = USERS[acting_user_id]
    project = PROJECTS[project_identifier]

    # Check permissions
    is_admin = acting_user.get("role") == "admin"
    is_owner = project.get("owner_id") == acting_user_id

    if not (is_admin or is_owner):
        return False  # User does not have permission

    project_members = project.get("members", [])

    if modification == 'add':
        if user_to_update_id not in project_members:
            project_members.append(user_to_update_id)
            PROJECTS[project_identifier]["members"] = project_members
        return True  # Successfully added or user was already a member
    
    elif modification == 'remove':
        # Project owner cannot be removed from their own project's member list
        if user_to_update_id == project.get("owner_id"):
            return False 

        if user_to_update_id in project_members:
            project_members.remove(user_to_update_id)
            PROJECTS[project_identifier]["members"] = project_members
        return True  # Successfully removed or user was not a member
    
    return False # Should not be reached if modification is 'add' or 'remove'
