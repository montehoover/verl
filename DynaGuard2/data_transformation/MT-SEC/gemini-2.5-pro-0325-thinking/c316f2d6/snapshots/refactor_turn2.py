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


def _has_permission(current_user_id: int, project_id: int, users_data: dict, projects_data: dict) -> bool:
    """Checks if the current user has permission to modify the project."""
    current_user = users_data[current_user_id]
    project = projects_data[project_id]
    is_admin = current_user["role"] == "admin"
    is_owner = project["owner_id"] == current_user_id
    return is_admin or is_owner


def _modify_project_members(project_id: int, action: str, target_user_id: int, projects_data: dict) -> bool:
    """Adds or removes a member from a project's member list."""
    project = projects_data[project_id]
    if action == 'add':
        if target_user_id not in project["members"]:
            project["members"].append(target_user_id)
            return True
        return False  # User is already a member
    elif action == 'remove':
        # Prevent removing the project owner
        if target_user_id == project["owner_id"]:
            return False
        if target_user_id in project["members"]:
            project["members"].remove(target_user_id)
            return True
        return False  # User is not a member
    return False # Invalid action


def manage_project_access(current_user_id: int, project_id: int, action: str, target_user_id: int) -> bool:
    """
    Manages access to a project by adding or removing members.

    Args:
        current_user_id: The ID of the user performing the action.
        project_id: The ID of the project to modify.
        action: The action to perform ('add' or 'remove').
        target_user_id: The ID of the user to add or remove.

    Returns:
        True if the action was successful, False otherwise.
    """
    if current_user_id not in USERS or target_user_id not in USERS:
        return False  # Actor or target user does not exist

    if project_id not in PROJECTS:
        return False  # Project does not exist

    if not _has_permission(current_user_id, project_id, USERS, PROJECTS):
        return False  # User is not authorized to modify this project

    return _modify_project_members(project_id, action, target_user_id, PROJECTS)
