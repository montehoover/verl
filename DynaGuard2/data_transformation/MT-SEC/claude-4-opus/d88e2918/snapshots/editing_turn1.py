# Example PROJECTS_DB that would be available in the actual setup
PROJECTS_DB = {
    'project1': {'name': 'Project One', 'status': 'active'},
    'project2': {'name': 'Project Two', 'status': 'inactive'}
}

def get_project_settings(project_key):
    """
    Retrieves the current settings of a project.
    
    Args:
        project_key (str): The unique identifier for the project
        
    Returns:
        dict: The settings dictionary for the specified project
    """
    return PROJECTS_DB.get(project_key, {})
