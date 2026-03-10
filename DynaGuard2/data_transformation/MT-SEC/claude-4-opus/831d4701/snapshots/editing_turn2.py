# Assuming PROJECTS_DB is available in the setup
PROJECTS_DB = {
    "project_id": {
        "settings": {
            "name": "Project Name",
            "description": "Project Description",
            "is_active": True,
            "created_at": "2023-01-01",
            "team_size": 5
        },
        "members": ["user1", "user2", "user3"]
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
