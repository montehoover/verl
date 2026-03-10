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


def adjust_project_params(user_token: str, role: str, project_ident: str, updated_values: dict) -> str:
    """
    Update project settings if the user is authorized.

    Args:
        user_token: Identifier representing the user attempting to update settings.
        role: User role (e.g., 'project_manager', 'team_member', 'viewer').
        project_ident: Identifier of the project being modified.
        updated_values: Dict containing new configuration for the project.

    Returns:
        A success message indicating that the settings have been updated.

    Raises:
        KeyError: If the project does not exist.
        TypeError: If updated_values is not a dictionary.
        PermissionError: If the user is not authorized to update settings.
    """
    if project_ident not in PROJECTS_DB:
        raise KeyError(f"Project '{project_ident}' not found")

    if not isinstance(updated_values, dict):
        raise TypeError("updated_values must be a dictionary")

    project = PROJECTS_DB[project_ident]

    # Authorization rules:
    # - User must be a member of the project
    # - Role must be 'project_manager' or 'team_member'
    allowed_roles = {"project_manager", "team_member"}
    is_member = user_token in project.get("members", [])
    is_allowed_role = role in allowed_roles

    if not (is_member and is_allowed_role):
        raise PermissionError("User is not authorized to update project settings")

    # Update settings
    project_settings = project.setdefault("settings", {})
    project_settings.update(updated_values)

    return f"Project settings for {project_ident} have been updated successfully"
