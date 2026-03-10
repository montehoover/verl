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
    Updates project settings if the user is authorized.

    Args:
        uid: The unique identifier of the user attempting the modification.
        role: The role of the user (e.g., 'project_manager', 'team_member', 'viewer').
        proj: The identifier of the project whose settings should be changed.
        settings_update: A dictionary containing the updated settings for the project.

    Returns:
        A success message upon successfully updating the project settings.

    Raises:
        ValueError: If the project ID is not found in PROJECTS_DB.
        PermissionError: If the user is not authorized to change project settings.
    """
    if proj not in PROJECTS_DB:
        raise ValueError(f"Project {proj} not found.")

    project_data = PROJECTS_DB[proj]
    authorized = False

    if role == 'project_manager':
        authorized = True
    elif role == 'team_member':
        if uid == project_data['creator_id']:
            authorized = True
    # Any other role (e.g., 'viewer') defaults to not authorized

    if authorized:
        project_data['settings'].update(settings_update)
        return f"Project settings for {proj} have been updated successfully"
    else:
        raise PermissionError("User does not have permission to change project settings.")
