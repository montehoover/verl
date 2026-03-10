import datetime

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

SUGGESTIONS_LOG = []

def adjust_project_params(user_token, role, project_ident, suggested_values):
    # Check if project exists
    if project_ident not in PROJECTS_DB:
        raise ValueError("Project not found")
    
    project = PROJECTS_DB[project_ident]
    
    # Check if user has access to this project
    if user_token not in project.get('members', []):
        raise PermissionError("User not a member of this project")
    
    # Define what fields each role can suggest changes to
    if role == 'project_manager':
        # Can suggest changes to all fields
        allowed_fields = {'project_id', 'creator_id', 'members', 'settings'}
    elif role == 'team_member':
        # Can only suggest changes to settings
        allowed_fields = {'settings'}
    elif role == 'viewer':
        # Cannot suggest any changes
        allowed_fields = set()
    else:
        raise PermissionError("Invalid role")
    
    # Check if viewer is trying to make suggestions
    if role == 'viewer' and suggested_values:
        raise PermissionError("Viewers cannot suggest project changes")
    
    # Validate suggested changes against allowed fields
    for field in suggested_values:
        if field not in allowed_fields:
            raise PermissionError(f"Role '{role}' cannot suggest changes to field: {field}")
    
    # If there are valid suggestions, log them
    if suggested_values:
        suggestion_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_token": user_token,
            "role": role,
            "project_id": project_ident,
            "suggested_values": suggested_values
        }
        SUGGESTIONS_LOG.append(suggestion_entry)
    
    # Return current project data (suggestions are logged but not applied)
    return project
