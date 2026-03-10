# Mock database for projects
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

def get_project_settings(project_id, user_id):
    """
    Retrieves the current settings of a project with role-based access check.
    
    Args:
        project_id (str): The ID of the project
        user_id (str): The ID of the user requesting access
        
    Returns:
        dict: The project settings if user has access, error message otherwise
    """
    if project_id not in PROJECTS_DB:
        return {"error": "Project not found"}
    
    project = PROJECTS_DB[project_id]
    members = project.get('members', [])
    
    if user_id not in members:
        return {"error": "Insufficient permissions: User is not a member of this project"}
    
    return project.get('settings', {})

def change_project_settings(owner_id, access_level, proj_id, modifications):
    """
    Allows authorized users to update project settings.
    
    Args:
        owner_id (str): The user attempting to update settings
        access_level (str): The user's role
        proj_id (str): The identifier of the project
        modifications (dict): The new configuration
        
    Returns:
        str: Success message if settings are updated
        
    Raises:
        PermissionError: If user is not authorized to update settings
    """
    if proj_id not in PROJECTS_DB:
        raise PermissionError("Project not found")
    
    project = PROJECTS_DB[proj_id]
    
    # Check if user is the creator or has admin access
    if owner_id != project['creator_id'] and access_level != 'admin':
        raise PermissionError("Unauthorized access: Only project creator or admin can modify settings")
    
    # Check if user is a member of the project
    if owner_id not in project['members']:
        raise PermissionError("Unauthorized access: User is not a member of this project")
    
    # Update the settings
    project['settings'].update(modifications)
    
    return "Project settings updated successfully"
