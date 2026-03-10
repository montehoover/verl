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

def _has_permission_to_update_settings(owner_id: str, access_level: str, project: dict) -> bool:
    """
    Checks if the user has permission to update project settings.

    Args:
        owner_id: The identifier of the user.
        access_level: The user's access level.
        project: The project dictionary.

    Returns:
        True if the user has permission, False otherwise.
    """
    is_creator = (owner_id == project["creator_id"])
    is_project_manager = (access_level == 'project_manager')
    return is_creator or is_project_manager

def _update_project_settings_in_db(project: dict, modifications: dict):
    """
    Updates the project settings in the database (in-memory dictionary).

    Args:
        project: The project dictionary to update.
        modifications: A dictionary containing the new configuration for the project.
    """
    project["settings"] = modifications

def change_project_settings(owner_id: str, access_level: str, proj_id: str, modifications: dict) -> str:
    """
    Allows authorized users to update project settings.

    Args:
        owner_id: The identifier representing the user attempting to update settings.
        access_level: The user's role (e.g., 'project_manager', 'team_member', 'viewer').
        proj_id: The identifier of the project being modified.
        modifications: A dictionary containing the new configuration for the project.

    Returns:
        A success message indicating that the settings have been updated.

    Raises:
        ValueError: If the project_id does not exist in PROJECTS_DB.
        PermissionError: If the user does not have sufficient rights to modify the project settings.
    """
    project = PROJECTS_DB.get(proj_id)

    if not project:
        raise ValueError(f"Project with ID '{proj_id}' not found.")

    if not _has_permission_to_update_settings(owner_id, access_level, project):
        raise PermissionError(f"User '{owner_id}' with access level '{access_level}' is not authorized to change settings for project '{proj_id}'.")

    _update_project_settings_in_db(project, modifications)
    return f"Settings for project '{proj_id}' updated successfully."
