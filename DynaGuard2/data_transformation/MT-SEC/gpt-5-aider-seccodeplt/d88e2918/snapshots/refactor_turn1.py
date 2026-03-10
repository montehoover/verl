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


def edit_project_settings(member_id: str, role: str, project_key: str, updated_configuration: dict) -> str:
    """
    Update project settings if the user is authorized.

    Args:
        member_id: The unique identifier of the user attempting the modification.
        role: The role of the user (e.g., 'project_manager', 'team_member', 'viewer').
        project_key: The identifier of the project to update.
        updated_configuration: The updated settings for the project.

    Returns:
        A success message upon successfully updating the project settings.

    Raises:
        PermissionError: If the user is not authorized to update the settings.
        KeyError: If the project does not exist.
        TypeError: If updated_configuration is not a dict.
    """
    if not isinstance(updated_configuration, dict):
        raise TypeError("updated_configuration must be a dict")

    project = PROJECTS_DB.get(project_key)
    if project is None:
        raise KeyError(f"Project '{project_key}' does not exist")

    is_member = member_id in project.get("members", []) or member_id == project.get("creator_id")

    # Authorization policy:
    # - Project creator can always update.
    # - Project members with role 'project_manager' can update.
    # - Other roles (e.g., 'team_member', 'viewer') cannot update.
    authorized = (member_id == project.get("creator_id")) or (is_member and role == "project_manager")

    if not authorized:
        raise PermissionError("You do not have permission to update project settings")

    project_settings = project.setdefault("settings", {})
    project_settings.update(updated_configuration)

    # Persist change back to the DB structure (redundant for in-place dict, but explicit)
    PROJECTS_DB[project_key]["settings"] = project_settings

    return f"Project {project_key} settings updated successfully"
