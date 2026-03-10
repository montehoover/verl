# Mock database for projects
PROJECTS_DB = {}

def get_project_settings(project_id):
    """
    Retrieves the current settings for a given project.
    
    Args:
        project_id (str): The ID of the project
        
    Returns:
        dict: The settings dictionary for the project
    """
    return PROJECTS_DB.get(project_id, {}).get('settings', {})
