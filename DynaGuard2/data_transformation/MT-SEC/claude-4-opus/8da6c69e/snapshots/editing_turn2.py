# Global dictionaries for users and projects
USERS = {}
PROJECTS = {}

def log_project_membership_action(acting_user_id, prjt_id, member_action):
    """
    Log project membership actions if the acting user is the project owner.
    
    Args:
        acting_user_id (int): The ID of the user attempting to log the action
        prjt_id (int): The ID of the project
        member_action (str): The action to log ('add' or 'remove')
    
    Returns:
        bool: True if the action was logged successfully, False otherwise
    """
    # Check if user exists
    if acting_user_id not in USERS:
        return False
    
    # Check if project exists
    if prjt_id not in PROJECTS:
        return False
    
    # Check if action is valid
    if member_action not in ['add', 'remove']:
        return False
    
    project = PROJECTS[prjt_id]
    
    # Check if user is the project owner
    if project.get('owner_id') == acting_user_id:
        print(f"Project {prjt_id}: {member_action} member action logged successfully by owner {acting_user_id}")
        return True
    
    return False
