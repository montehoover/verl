# Global dictionaries for users and projects
USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

PROJECTS = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}

def handle_project_membership(acting_user_id, prjt_id, member_action, target_member_id):
    """
    Handle adding or removing members from projects.
    
    Args:
        acting_user_id (int): The ID of the user executing the action
        prjt_id (int): The ID of the project being updated
        member_action (str): The action to perform ('add' or 'remove')
        target_member_id (int): The ID of the member being targeted by the action
    
    Returns:
        bool: True if the operation was successfully completed, False otherwise
    """
    # Check if acting user exists
    if acting_user_id not in USERS:
        return False
    
    # Check if target member exists
    if target_member_id not in USERS:
        return False
    
    # Check if project exists
    if prjt_id not in PROJECTS:
        return False
    
    # Check if action is valid
    if member_action not in ['add', 'remove']:
        return False
    
    user = USERS[acting_user_id]
    project = PROJECTS[prjt_id]
    
    # Check if user is admin or project owner
    is_admin = user.get('role') == 'admin'
    is_owner = project.get('owner_id') == acting_user_id
    
    if not (is_admin or is_owner):
        return False
    
    # Perform the action
    if member_action == 'add':
        if target_member_id not in project['members']:
            project['members'].append(target_member_id)
            return True
        else:
            # Member already in project
            return False
    elif member_action == 'remove':
        if target_member_id in project['members']:
            project['members'].remove(target_member_id)
            return True
        else:
            # Member not in project
            return False
    
    return False
