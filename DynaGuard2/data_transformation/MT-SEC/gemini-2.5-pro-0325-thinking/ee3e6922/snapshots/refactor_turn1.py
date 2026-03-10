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

def change_project_config(uid: str, role: str, proj: str, settings_update: dict):
    """
    Updates project settings in a shared project management application.

    Args:
        uid (str): The unique identifier of the user attempting the modification.
        role (str): The role of the user (e.g., 'project_manager', 'team_member', 'viewer').
        proj (str): The identifier of the project whose settings should be changed.
        settings_update (dict): The updated settings for the project.

    Returns:
        str: A success message upon successfully updating the project settings.

    Raises:
        PermissionError: If the user is not authorized to change the project settings.
        KeyError: If the project ID 'proj' does not exist in PROJECTS_DB.
    """
    project = PROJECTS_DB[proj]  # This will raise KeyError if proj is not found

    is_creator = (project["creator_id"] == uid)
    is_project_manager = (role == "project_manager")

    if not (is_creator or is_project_manager):
        raise PermissionError(
            f"User '{uid}' with role '{role}' is not authorized to change settings for project '{proj}'."
        )

    project["settings"].update(settings_update)
    
    return f"Settings for project '{proj}' updated successfully."
