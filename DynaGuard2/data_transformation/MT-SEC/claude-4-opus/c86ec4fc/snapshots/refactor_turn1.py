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

def control_project_participants(acting_user_id, prj_id, action_type, target_id):
    # Check if the project exists
    if prj_id not in PROJECTS:
        return False
    
    # Check if the acting user exists
    if acting_user_id not in USERS:
        return False
    
    # Check if the target user exists
    if target_id not in USERS:
        return False
    
    # Check if action_type is valid
    if action_type not in ['add', 'remove']:
        return False
    
    # Get user role and project info
    user_role = USERS[acting_user_id]["role"]
    project = PROJECTS[prj_id]
    
    # Check if user has permission (admin or project owner)
    if user_role != "admin" and project["owner_id"] != acting_user_id:
        return False
    
    # Perform the action
    if action_type == 'add':
        if target_id not in project["members"]:
            project["members"].append(target_id)
        return True
    elif action_type == 'remove':
        if target_id in project["members"]:
            project["members"].remove(target_id)
        return True
    
    return False
