# Global data structures
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

def project_access_control(active_user_id, prj_id, action_type, member_user_id):
    """Allow project owners to add/remove team members from their projects, admins can manage any project.
    
    Args:
        active_user_id (int): The user performing the task.
        prj_id (int): The project ID.
        action_type (str): Either 'add' or 'remove'.
        member_user_id (int): The user affected by the change.
        
    Returns:
        bool: True if modification is successful, False otherwise.
    """
    # Check if project exists
    if prj_id not in PROJECTS:
        return False
    
    # Check if active user exists
    if active_user_id not in USERS:
        return False
    
    # Check if member user exists
    if member_user_id not in USERS:
        return False
    
    # Check if action type is valid
    if action_type not in ['add', 'remove']:
        return False
    
    # Check permissions
    is_admin = USERS[active_user_id]["role"] == "admin"
    is_owner = PROJECTS[prj_id]["owner_id"] == active_user_id
    
    if not (is_admin or is_owner):
        return False
    
    # Perform the action
    if action_type == 'add':
        if member_user_id not in PROJECTS[prj_id]["members"]:
            PROJECTS[prj_id]["members"].append(member_user_id)
            return True
        return False
    else:  # action_type == 'remove'
        if member_user_id in PROJECTS[prj_id]["members"]:
            PROJECTS[prj_id]["members"].remove(member_user_id)
            return True
        return False
