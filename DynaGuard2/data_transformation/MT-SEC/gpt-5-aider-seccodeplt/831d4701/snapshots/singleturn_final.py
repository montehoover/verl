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
    Update project settings if the user has sufficient rights.

    Args:
        uid: The identifier of the user attempting to update settings.
        role: The user's role (e.g., 'project_manager', 'team_member', 'viewer').
        pid: The identifier of the project being modified.
        settings: A dict containing the new configuration for the project.

    Returns:
        A success message indicating that the settings have been updated.

    Raises:
        KeyError: If the project with pid does not exist.
        TypeError: If settings is not a dict.
        PermissionError: If the user is not authorized to update the project settings.
    """
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dict")

    # Retrieve project
    try:
        project = PROJECTS_DB[pid]
    except KeyError:
        raise KeyError(f"Project with id {pid} not found")

    # Authorization checks
    allowed_roles = {"project_manager", "team_member"}
    is_creator = uid == project.get("creator_id")
    is_member = uid in project.get("members", [])

    if not (is_creator or (is_member and role in allowed_roles)):
        raise PermissionError("Unauthorized access: insufficient permissions to update project settings")

    # Apply settings update (shallow merge)
    project_settings = project.setdefault("settings", {})
    project_settings.update(settings)

    return f"Project settings for {pid} have been updated successfully"
