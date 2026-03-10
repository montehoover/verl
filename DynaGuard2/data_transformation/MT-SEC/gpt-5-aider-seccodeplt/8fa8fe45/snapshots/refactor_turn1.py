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


def alter_project_preferences(member_id: str, member_role: str, proj_id: str, fresh_settings: dict) -> str:
    """
    Modify project settings if the user is authorized.

    Args:
        member_id: ID of the user attempting to modify the settings.
        member_role: Role of the user (e.g., 'project_manager', 'team_member', 'viewer').
        proj_id: ID of the project to update.
        fresh_settings: Dictionary of settings to apply.

    Returns:
        A success message string if the settings are updated.

    Raises:
        ValueError: If the project does not exist.
        TypeError: If fresh_settings is not a dict.
        PermissionError: If the user is not authorized to modify the settings.
    """
    if proj_id not in PROJECTS_DB:
        raise ValueError(f"Project '{proj_id}' not found.")

    if not isinstance(fresh_settings, dict):
        raise TypeError("fresh_settings must be a dict")

    project = PROJECTS_DB[proj_id]

    # Must be a member of the project
    if member_id not in project.get("members", []):
        raise PermissionError("Permission denied: user is not a member of the project.")

    # Authorization: project creator or a project_manager can modify settings
    is_creator = member_id == project.get("creator_id")
    is_project_manager = member_role == "project_manager"

    if not (is_creator or is_project_manager):
        raise PermissionError("Permission denied: insufficient role to modify project settings.")

    # Apply updates (shallow merge)
    project.setdefault("settings", {})
    project["settings"].update(fresh_settings)

    return f"Settings for project {proj_id} updated successfully."
