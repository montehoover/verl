# Global dictionaries for users and projects
USERS = {}
PROJECTS = {}

def manage_project_access(acting_user_id, prjt_id):
    """
    Verify if the acting user can manage the specified project.
    
    Args:
        acting_user_id (int): The ID of the user attempting to manage the project
        prjt_id (int): The ID of the project to be managed
    
    Returns:
        bool: True if the user has access to manage the project, False otherwise
    """
    # Check if user exists
    if acting_user_id not in USERS:
        return False
    
    # Check if project exists
    if prjt_id not in PROJECTS:
        return False
    
    user = USERS[acting_user_id]
    project = PROJECTS[prjt_id]
    
    # Check if user is an admin or the project owner
    if user.get('is_admin', False) or project.get('owner_id') == acting_user_id:
        print(f"User {acting_user_id} has access to manage project {prjt_id}")
        return True
    
    return False
