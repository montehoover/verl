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

def adjust_project_params(user_token, role, project_ident, updated_values):
    # Check if project exists
    if project_ident not in PROJECTS_DB:
        raise ValueError("Project not found")
    
    project = PROJECTS_DB[project_ident]
    
    # Check if user has access to this project
    if user_token not in project.get('members', []):
        raise PermissionError("User not a member of this project")
    
    # Check authorization - only project_managers or the creator can update
    is_creator = user_token == project.get('creator_id')
    is_project_manager = role == 'project_manager'
    
    if not (is_creator or is_project_manager):
        raise PermissionError("Only project managers or the project creator can update project settings")
    
    # Apply the updates to the project
    for field, value in updated_values.items():
        if field in project:
            if isinstance(project[field], dict) and isinstance(value, dict):
                # For nested dicts like settings, update the nested values
                project[field].update(value)
            else:
                # For other fields, replace the value
                project[field] = value
        else:
            # Add new field if it doesn't exist
            project[field] = value
    
    return "Project updated successfully"
