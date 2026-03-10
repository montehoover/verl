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
    # Check if project exists
    if project_ident not in PROJECTS_DB:
        raise PermissionError("Project not found")
    
    project = PROJECTS_DB[project_ident]
    
    # Check if user is authorized based on role and membership
    if role == 'viewer':
        raise PermissionError("Viewers do not have permission to update project settings")
    
    if role in ['project_manager', 'team_member']:
        # Check if user is either the creator or a member of the project
        if user_token != project['creator_id'] and user_token not in project['members']:
            raise PermissionError("User is not authorized to update this project")
        
        # Update the project settings
        project['settings'].update(updated_values)
        return f"Project {project_ident} settings have been successfully updated"
    
    # If role is not recognized, raise error
    raise PermissionError("Invalid user role")
