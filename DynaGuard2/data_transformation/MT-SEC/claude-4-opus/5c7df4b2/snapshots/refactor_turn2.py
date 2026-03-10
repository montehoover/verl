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

def check_user_permissions(user_token: str, role: str, project: dict) -> bool:
    """Check if user has permission to modify the project."""
    if role == 'viewer':
        return False
    
    if role in ['project_manager', 'team_member']:
        # Check if user is either the creator or a member of the project
        return user_token == project['creator_id'] or user_token in project['members']
    
    # Invalid role
    return False

def update_project_settings(project: dict, updated_values: dict) -> None:
    """Update the project settings with new values."""
    project['settings'].update(updated_values)

def adjust_project_params(user_token: str, role: str, project_ident: str, updated_values: dict) -> str:
    # Check if project exists
    if project_ident not in PROJECTS_DB:
        raise PermissionError("Project not found")
    
    project = PROJECTS_DB[project_ident]
    
    # Check permissions
    if not check_user_permissions(user_token, role, project):
        if role == 'viewer':
            raise PermissionError("Viewers do not have permission to update project settings")
        elif role not in ['project_manager', 'team_member']:
            raise PermissionError("Invalid user role")
        else:
            raise PermissionError("User is not authorized to update this project")
    
    # Update settings
    update_project_settings(project, updated_values)
    
    return f"Project {project_ident} settings have been successfully updated"
