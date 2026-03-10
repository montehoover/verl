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


def change_project_settings(owner_id: str, access_level: str, proj_id: str, modifications: dict) -> str:
    """
    Update project settings if the user is authorized.

    Args:
        owner_id (str): Identifier of the user attempting the update.
        access_level (str): User's role ('project_manager', 'team_member', 'viewer', etc.).
        proj_id (str): Identifier of the project to be modified.
        modifications (dict): New settings to apply to the project.

    Returns:
        str: Success message indicating that the settings have been updated.

    Raises:
        KeyError: If the project does not exist.
        PermissionError: If the user is not authorized to update the settings.
    """
    project = PROJECTS_DB.get(proj_id)
    if project is None:
        raise KeyError(f"Project '{proj_id}' not found")

    # User must be a member of the project
    if owner_id not in project.get("members", []):
        raise PermissionError("Unauthorized access: user is not a member of this project")

    # Only certain roles can update settings
    allowed_roles = {"project_manager", "team_member"}
    if access_level not in allowed_roles:
        raise PermissionError("Unauthorized access: insufficient privileges to update project settings")

    # Apply modifications to the project's settings
    project_settings = project.setdefault("settings", {})
    if not isinstance(modifications, dict):
        raise ValueError("modifications must be a dictionary")

    project_settings.update(modifications)

    return f"Project settings for {proj_id} have been updated successfully"
