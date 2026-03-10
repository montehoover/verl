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
    Enables a user to modify the settings for a project.

    Args:
        member_id: ID of the user attempting to modify settings.
        member_role: Role of the user (e.g., 'project_manager', 'team_member', 'viewer').
        proj_id: ID of the project to be updated.
        fresh_settings: Dictionary containing the updated settings.

    Returns:
        A string indicating successful update.

    Raises:
        ValueError: If the project ID does not exist.
        PermissionError: For unauthorized access.
    """
    project = PROJECTS_DB.get(proj_id)
    if not project:
        raise ValueError(f"Project with ID '{proj_id}' not found.")

    is_authorized = False
    if member_role == 'project_manager':
        is_authorized = True
    elif member_role == 'team_member':
        if member_id == project['creator_id']:
            is_authorized = True
    # Other roles like 'viewer' or any unspecified role will default to not authorized

    if not is_authorized:
        raise PermissionError(f"User {member_id} with role {member_role} does not have permission to alter settings for project {proj_id}.")

    # Update the project settings
    project['settings'].update(fresh_settings)
    
    return f"Project settings for {proj_id} have been updated successfully"
