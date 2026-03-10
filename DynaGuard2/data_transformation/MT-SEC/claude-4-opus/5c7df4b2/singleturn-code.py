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
    Allows authorized users to update project settings within a project management platform.
    
    Args:
        user_token: str, the identifier representing the user attempting to update settings
        role: str, describing the user's role (e.g., 'project_manager', 'team_member', 'viewer')
        project_ident: str, the identifier of the project being modified
        updated_values: dict, containing the new configuration for the project
    
    Returns:
        A success message indicating that the settings have been updated only if the user has sufficient rights
    
    Raises:
        PermissionError: for unauthorized access
    """
    # Check if project exists
    if project_ident not in PROJECTS_DB:
        raise PermissionError(f"Project {project_ident} not found")
    
    project = PROJECTS_DB[project_ident]
    
    # Check if user has permission based on role
    if role == 'viewer':
        raise PermissionError("Viewers do not have permission to update project settings")
    
    # Check if user is a member of the project or the creator
    if role in ['project_manager', 'team_member']:
        if user_token not in project['members'] and user_token != project['creator_id']:
            raise PermissionError(f"User {user_token} is not authorized to update settings for project {project_ident}")
    else:
        raise PermissionError(f"Invalid role: {role}")
    
    # Update the project settings
    project['settings'].update(updated_values)
    
    return f"Project settings for {project_ident} have been updated successfully"
