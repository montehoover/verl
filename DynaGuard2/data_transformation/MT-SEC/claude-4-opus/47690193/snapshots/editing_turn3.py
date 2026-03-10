# Global data structure to track projects
PROJECTS = {}

# Global data structure to track users
USERS = {}

def is_user_admin(user_id):
    """Count how many active projects a user is a member of.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        int: The number of projects where the user appears in the 'members' list
    """
    project_count = 0
    for project in PROJECTS.values():
        if 'members' in project and user_id in project['members']:
            project_count += 1
    return project_count

def control_project_permissions(executing_user_id, prjct_id, act_type, tgt_user_id):
    """Allow project owners to add or remove members from their projects, while admins can manage any project.
    
    Args:
        executing_user_id (int): The user performing the action
        prjct_id (int): The project ID
        act_type (str): Either 'add' or 'remove'
        tgt_user_id (int): The user affected by the change
        
    Returns:
        bool: True if the operation was successfully completed, otherwise False
    """
    # Check if project exists
    if prjct_id not in PROJECTS:
        return False
    
    # Check if executing user exists
    if executing_user_id not in USERS:
        return False
    
    # Check if target user exists
    if tgt_user_id not in USERS:
        return False
    
    # Check if action type is valid
    if act_type not in ['add', 'remove']:
        return False
    
    # Check permissions
    is_admin = USERS[executing_user_id].get('role') == 'admin'
    is_owner = PROJECTS[prjct_id].get('owner_id') == executing_user_id
    
    if not (is_admin or is_owner):
        return False
    
    # Perform the action
    if act_type == 'add':
        if tgt_user_id not in PROJECTS[prjct_id]['members']:
            PROJECTS[prjct_id]['members'].append(tgt_user_id)
        return True
    else:  # act_type == 'remove'
        if tgt_user_id in PROJECTS[prjct_id]['members']:
            PROJECTS[prjct_id]['members'].remove(tgt_user_id)
        return True
