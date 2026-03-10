# Mock database for projects
PROJECTS_DB = {}

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
