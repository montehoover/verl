# Example PROJECTS_DB that would be available in the actual setup
PROJECTS_DB = {
    'project1': {
        'name': 'Project One', 
        'status': 'active',
        'members': {
            'user1': {'role': 'admin'},
            'user2': {'role': 'viewer'},
            'user3': {'role': 'editor'}
        }
    },
    'project2': {
        'name': 'Project Two', 
        'status': 'inactive',
        'members': {
            'user1': {'role': 'viewer'},
            'user4': {'role': 'admin'}
        }
    }
}

# Roles that can view or modify settings
ALLOWED_ROLES = {'admin', 'editor'}

def get_project_settings(project_key):
    """
    Retrieves the current settings of a project.
    
    Args:
        project_key (str): The unique identifier for the project
        
    Returns:
        dict: The settings dictionary for the specified project
    """
    return PROJECTS_DB.get(project_key, {})

def check_user_role(member_id, project_key):
    """
    Checks if a user has permission to view or modify project settings.
    
    Args:
        member_id (str): The unique identifier for the member
        project_key (str): The unique identifier for the project
        
    Returns:
        bool: True if user is a member with allowed role, False otherwise
    """
    project = PROJECTS_DB.get(project_key, {})
    members = project.get('members', {})
    
    if member_id in members:
        user_role = members[member_id].get('role', '')
        return user_role in ALLOWED_ROLES
    
    return False
