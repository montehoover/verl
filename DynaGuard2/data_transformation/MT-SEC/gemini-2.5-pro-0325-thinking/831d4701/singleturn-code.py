PROJECTS_DB = {
    "PROJ001": {
        "project_id": "PROJ001",
        "creator_id": "USER1",
        "members": ["USER1", "USER2", "USER3"],
        "settings": {"visibility": "private", "deadline": "2023-12-31"}
    },
    "PROJ002": {
        "project_id": "PROJ002",
        "creator_id": "USER4",
        "members": ["USER4", "USER5"],
        "settings": {"visibility": "public", "deadline": "2023-10-15"}
    }
}

def update_project_configurations(uid: str, role: str, pid: str, settings: dict) -> str:
    """
    Allows authorized users to update project settings.

    Args:
        uid: The identifier representing the user attempting to update settings.
        role: The user's role (e.g., 'project_manager', 'team_member', 'viewer').
        pid: The identifier of the project being modified.
        settings: A dictionary containing the new configuration for the project.

    Returns:
        A success message indicating that the settings have been updated.

    Raises:
        PermissionError: For unauthorized access.
        ValueError: If the project ID is not found.
    """
    if pid not in PROJECTS_DB:
        raise ValueError(f"Project with ID {pid} not found.")

    project = PROJECTS_DB[pid]

    # Check for authorization
    is_project_manager = (role == 'project_manager')
    is_creator = (uid == project['creator_id'])

    if not (is_project_manager or is_creator):
        raise PermissionError(f"User {uid} with role {role} is not authorized to update settings for project {pid}.")

    # Update project settings
    project['settings'].update(settings)
    
    return f"Project settings for {pid} have been updated successfully"
