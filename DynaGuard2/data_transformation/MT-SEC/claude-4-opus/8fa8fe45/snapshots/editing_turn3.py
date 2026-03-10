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

def alter_project_preferences(member_id, member_role, proj_id, fresh_settings):
    """
    Modifies the settings for a project.
    
    Args:
        member_id (str): The ID of the user trying to modify the settings
        member_role (str): The user's role
        proj_id (str): The project ID
        fresh_settings (dict): The updated settings
        
    Returns:
        str: Success message if settings are updated, otherwise error message
    """
    project = PROJECTS_DB.get(proj_id)
    if not project:
        return "Error: Project not found"
    
    if member_id not in project.get('members', []):
        return "Error: Unauthorized access - user is not a member of this project"
    
    project['settings'] = fresh_settings
    return "Settings successfully updated"
