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

def change_project_config(uid: str, role: str, proj: str, settings_update: dict) -> str:
    # Check if project exists
    if proj not in PROJECTS_DB:
        raise PermissionError(f"Project {proj} not found")
    
    project = PROJECTS_DB[proj]
    
    # Check authorization based on role and membership
    if role == 'viewer':
        # Viewers cannot modify project settings
        raise PermissionError("Viewers are not authorized to modify project settings")
    
    if role in ['project_manager', 'team_member']:
        # Check if user is a member of the project
        if uid not in project['members']:
            raise PermissionError(f"User {uid} is not a member of project {proj}")
        
        # Only project creator (assumed to be project manager) can change settings
        # Team members can only change settings if they are the creator
        if role == 'team_member' and uid != project['creator_id']:
            raise PermissionError("Only project managers can modify project settings")
        
        # Update the project settings
        project['settings'].update(settings_update)
        
        return f"Successfully updated settings for project {proj}"
    
    # Unknown role
    raise PermissionError(f"Unknown role: {role}")
