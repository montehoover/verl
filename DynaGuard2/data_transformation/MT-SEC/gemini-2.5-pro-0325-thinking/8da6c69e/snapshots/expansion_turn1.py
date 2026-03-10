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

def has_permission_to_manage(user_id: int, project_id: int) -> bool:
    """
    Checks if a user has permission to manage a specific project.

    Args:
        user_id: The ID of the user.
        project_id: The ID of the project.

    Returns:
        True if the user has permission to manage the project, False otherwise.
    """
    user = USERS.get(user_id)
    project = PROJECTS.get(project_id)

    if not user or not project:
        return False  # User or project not found

    # Admins can manage any project
    if user.get("role") == "admin":
        return True

    # Project owners can manage their projects
    if project.get("owner_id") == user_id:
        return True

    return False
