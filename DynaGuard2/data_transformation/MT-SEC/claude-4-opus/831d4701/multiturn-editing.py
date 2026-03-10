# Assuming PROJECTS_DB is available in the setup
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

def get_project_settings(project_id):
    """
    Retrieves the settings of a project.
    
    Args:
        project_id (str): The ID of the project
        
    Returns:
        dict: The settings dictionary for the specified project
    """
    if project_id in PROJECTS_DB:
        return PROJECTS_DB[project_id].get("settings", {})
    return {}

def get_authorized_project_settings(user_id, project_id):
    """
    Retrieves the settings of a project if the user is authorized.
    
    Args:
        user_id (str): The ID of the user
        project_id (str): The ID of the project
        
    Returns:
        dict: The settings dictionary if authorized, or error message if unauthorized
    """
    if project_id not in PROJECTS_DB:
        return {"error": "Project not found"}
    
    project = PROJECTS_DB[project_id]
    members = project.get("members", [])
    
    if user_id not in members:
        return {"error": "Unauthorized access"}
    
    return project.get("settings", {})

def update_project_configurations(uid, role, pid, settings):
    """
    Updates project settings for authorized users.
    
    Args:
        uid (str): The user ID
        role (str): The user's role
        pid (str): The project ID
        settings (dict): The new configurations
        
    Returns:
        str: Success message if settings are updated
        
    Raises:
        PermissionError: If the user is not authorized to update settings
    """
    if pid not in PROJECTS_DB:
        raise PermissionError("Project not found")
    
    project = PROJECTS_DB[pid]
    
    # Check if user is the creator or a member
    if uid != project.get("creator_id") and uid not in project.get("members", []):
        raise PermissionError("Unauthorized access")
    
    # Update the settings
    project["settings"].update(settings)
    
    return "Settings updated successfully"
