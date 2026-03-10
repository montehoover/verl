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
    Updates project settings in a shared project management application.

    Args:
        member_id: The unique identifier of the user attempting the modification.
        role: The role of the user (e.g., 'project_manager', 'team_member', 'viewer').
        project_key: The identifier of the project whose settings should be changed.
        updated_configuration: The updated settings for the project.

    Returns:
        A success message upon successfully updating the project settings.

    Raises:
        ValueError: If the project_key does not exist in PROJECTS_DB.
        PermissionError: For an unauthorized access.
    """
    if project_key not in PROJECTS_DB:
        raise ValueError(f"Project with key '{project_key}' not found.")

    project = PROJECTS_DB[project_key]

    # Authorization check:
    # User is authorized if they are a 'project_manager' or the project's creator.
    is_authorized = (role == 'project_manager') or (member_id == project['creator_id'])

    if not is_authorized:
        raise PermissionError("User not authorized to edit project settings.")

    # Update project settings
    project['settings'].update(updated_configuration)
    
    return "Project settings updated successfully."
