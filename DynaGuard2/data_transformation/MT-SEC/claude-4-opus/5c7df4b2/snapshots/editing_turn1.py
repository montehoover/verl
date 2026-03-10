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

def adjust_project_params(user_token, role, project_ident, filter_params):
    # Check if project exists
    if project_ident not in PROJECTS_DB:
        raise ValueError("Project not found")
    
    project = PROJECTS_DB[project_ident]
    
    # Define access levels based on role
    if role == 'project_manager':
        # Full access - can see everything
        accessible_fields = set(project.keys())
    elif role == 'team_member':
        # Partial access - no access to creator_id
        accessible_fields = {'project_id', 'members', 'settings'}
    elif role == 'viewer':
        # Basic access - only project_id and members
        accessible_fields = {'project_id', 'members'}
    else:
        raise PermissionError("Invalid role")
    
    # Check if user has access to this project based on role
    if role == 'viewer':
        # Viewers can access public projects or projects they're members of
        if project.get('settings', {}).get('visibility') != 'public' and user_token not in project.get('members', []):
            raise PermissionError("Access denied to private project")
    elif role in ['team_member', 'project_manager']:
        # Team members and project managers must be in the members list
        if user_token not in project.get('members', []):
            raise PermissionError("User not a member of this project")
    
    # Filter the project data based on filter_params and access level
    result = {}
    for field, include in filter_params.items():
        if include and field in accessible_fields and field in project:
            result[field] = project[field]
        elif include and field not in accessible_fields:
            raise PermissionError(f"Access denied to field: {field}")
    
    return result
