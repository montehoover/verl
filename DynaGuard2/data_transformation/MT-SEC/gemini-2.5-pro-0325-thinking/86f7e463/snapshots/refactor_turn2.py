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


def _has_permission(acting_user_id: int, project_identifier: int, users_data: dict, projects_data: dict) -> bool:
    """
    Checks if the acting user has permission to modify the project.
    Assumes acting_user_id and project_identifier are valid keys in users_data and projects_data.
    """
    acting_user = users_data[acting_user_id]
    project = projects_data[project_identifier]

    is_admin = acting_user.get("role") == "admin"
    is_owner = project.get("owner_id") == acting_user_id
    return is_admin or is_owner


def _modify_members(project_identifier: int, modification: str, user_to_update_id: int, projects_data: dict, users_data: dict) -> list | None:
    """
    Modifies the member list for a project based on the given modification.
    Returns the new list of members if the modification is valid (even if no change occurred),
    otherwise None (e.g., trying to remove an owner).
    Assumes project_identifier and user_to_update_id are valid keys, and modification is 'add' or 'remove'.
    This function does not modify projects_data directly; it returns the proposed new member list.
    """
    project = projects_data[project_identifier]
    # user_to_update_id is assumed to be a valid key in users_data, validated by the caller.

    current_members = list(project.get("members", [])) # Work on a copy

    if modification == 'add':
        if user_to_update_id not in current_members:
            current_members.append(user_to_update_id)
        return current_members
    
    elif modification == 'remove':
        # Project owner cannot be removed from their own project's member list
        if user_to_update_id == project.get("owner_id"):
            return None 

        if user_to_update_id in current_members:
            current_members.remove(user_to_update_id)
        # If user_to_update_id was not in current_members, list is unchanged; operation is still considered successful.
        return current_members
    
    # This case should not be reached if 'modification' is pre-validated by the caller.
    return None 


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

    # Check permissions using helper function
    if not _has_permission(acting_user_id, project_identifier, USERS, PROJECTS):
        return False  # User does not have permission

    # Perform member modification using helper function
    # Pass USERS for completeness, though user_to_update_id is already validated.
    updated_member_list = _modify_members(project_identifier, modification, user_to_update_id, PROJECTS, USERS)

    if updated_member_list is None:
        # This indicates an invalid operation, e.g., trying to remove the project owner.
        return False
    
    # If modification was valid (even if it resulted in no change),
    # update the global PROJECTS dictionary with the new list of members.
    PROJECTS[project_identifier]["members"] = updated_member_list
    return True
