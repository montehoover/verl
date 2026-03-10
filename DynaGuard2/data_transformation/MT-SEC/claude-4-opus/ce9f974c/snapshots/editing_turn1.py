# Mock database for projects
PROJECTS_DB = {}

def get_project_settings(project_id):
    """
    Retrieves the current settings of a project.
    
    Args:
        project_id (str): The ID of the project
        
    Returns:
        dict: The project settings
    """
    if project_id in PROJECTS_DB:
        return PROJECTS_DB[project_id].get('settings', {})
    return {}
