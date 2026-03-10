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


def _has_permission(active_user_id: int, prj_id: int, users_data: dict, projects_data: dict) -> bool:
    """Checks if the active user has permission to manage the project."""
    active_user = users_data[active_user_id]
    project = projects_data[prj_id]
    is_admin = active_user["role"] == "admin"
    is_owner = project["owner_id"] == active_user_id
    return is_admin or is_owner


def _update_project_members(project: dict, action_type: str, member_user_id: int) -> bool:
    """Adds or removes a member from the project."""
    if action_type == "add":
        if member_user_id not in project["members"]:
            project["members"].append(member_user_id)
        return True
    elif action_type == "remove":
        if member_user_id == project["owner_id"]:
            return False  # Cannot remove project owner
        if member_user_id in project["members"]:
            project["members"].remove(member_user_id)
            return True
        else:
            return False  # Member not in project, cannot remove
    return False # Invalid action type


def project_access_control(active_user_id: int, prj_id: int, action_type: str, member_user_id: int) -> bool:
    """
    Manages project membership based on user roles and ownership.

    Args:
        active_user_id: The ID of the user performing the action.
        prj_id: The ID of the project being modified.
        action_type: 'add' or 'remove'.
        member_user_id: The ID of the user to be added or removed.

    Returns:
        True if the operation was successful, False otherwise.
    """
    if active_user_id not in USERS:
        return False  # Active user does not exist

    if prj_id not in PROJECTS:
        return False  # Project does not exist

    if member_user_id not in USERS:
        return False  # Member user does not exist

    if not _has_permission(active_user_id, prj_id, USERS, PROJECTS):
        return False  # Not authorized

    project = PROJECTS[prj_id]
    return _update_project_members(project, action_type, member_user_id)
