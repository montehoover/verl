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

    is_creator = (owner_id == project["creator_id"])
    is_project_manager = (access_level == 'project_manager')

    if not (is_creator or is_project_manager):
        raise PermissionError(f"User '{owner_id}' with access level '{access_level}' is not authorized to change settings for project '{proj_id}'.")

    project["settings"] = modifications
    return f"Settings for project '{proj_id}' updated successfully."
