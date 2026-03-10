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

def is_member_of_project(user_id, project_id):
    """
    Checks if a user is part of a project.
    
    Args:
        user_id (str): The ID of the user
        project_id (str): The ID of the project
        
    Returns:
        bool: True if the user is a member of the project, otherwise False
    """
    project = PROJECTS_DB.get(project_id, {})
    members = project.get('members', [])
    return user_id in members
