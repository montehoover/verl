# Assuming PROJECTS_DB is available in the setup
PROJECTS_DB = {
    "project_id": {
        "settings": {
            "name": "Project Name",
            "description": "Project Description",
            "is_active": True,
            "created_at": "2023-01-01",
            "team_size": 5
        }
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
