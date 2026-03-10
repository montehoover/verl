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

def handle_project_membership(acting_user_id: int, prjt_id: int, member_action: str, target_member_id: int) -> bool:
    # Check if project exists
    if prjt_id not in PROJECTS:
        return False
    
    # Check if acting user exists
    if acting_user_id not in USERS:
        return False
    
    # Check if target member exists
    if target_member_id not in USERS:
        return False
    
    # Check if member_action is valid
    if member_action not in ['add', 'remove']:
        return False
    
    # Get user role and project details
    acting_user = USERS[acting_user_id]
    project = PROJECTS[prjt_id]
    
    # Check if user has permission (admin or project owner)
    if acting_user['role'] != 'admin' and acting_user_id != project['owner_id']:
        return False
    
    # Perform the action
    if member_action == 'add':
        if target_member_id not in project['members']:
            project['members'].append(target_member_id)
            return True
        else:
            return False
    else:  # member_action == 'remove'
        if target_member_id in project['members']:
            project['members'].remove(target_member_id)
            return True
        else:
            return False
