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


def is_authorized_to_edit(member_id: str, role: str, project: dict) -> bool:
    """
    Pure function to determine if a user is authorized to edit project settings.

    Authorization policy:
    - Project creator can always update.
    - Project members with role 'project_manager' can update.
    - Other roles (e.g., 'team_member', 'viewer') cannot update.

    Args:
        member_id: The unique identifier of the user attempting the modification.
        role: The role of the user (e.g., 'project_manager', 'team_member', 'viewer').
        project: The project dictionary from the PROJECTS_DB.

    Returns:
        True if authorized, False otherwise.
    """
    creator_id = project.get("creator_id")
    members = project.get("members", [])
    is_member = member_id in members or member_id == creator_id
    return (member_id == creator_id) or (is_member and role == "project_manager")


def apply_settings_update(current_settings: dict, updated_configuration: dict) -> dict:
    """
    Pure function to compute the updated settings without mutating inputs.

    Performs a shallow merge where keys in updated_configuration override
    those in current_settings.

    Args:
        current_settings: Existing project settings.
        updated_configuration: New settings to apply.

    Returns:
        A new dict representing the merged settings.

    Raises:
        TypeError: If either argument is not a dict.
    """
    if not isinstance(current_settings, dict):
        raise TypeError("current_settings must be a dict")
    if not isinstance(updated_configuration, dict):
        raise TypeError("updated_configuration must be a dict")

    merged = current_settings.copy()
    merged.update(updated_configuration)
    return merged


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

    if not is_authorized_to_edit(member_id, role, project):
        raise PermissionError("You do not have permission to update project settings")

    current_settings = project.get("settings", {})
    new_settings = apply_settings_update(current_settings, updated_configuration)

    # Persist changes
    PROJECTS_DB[project_key]["settings"] = new_settings

    return f"Project {project_key} settings updated successfully"
