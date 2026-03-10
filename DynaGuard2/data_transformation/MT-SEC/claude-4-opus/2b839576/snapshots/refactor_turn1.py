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

def project_access_control(active_user_id: int, prj_id: int, action_type: str, member_user_id: int) -> bool:
    # Check if the project exists
    if prj_id not in PROJECTS:
        return False
    
    # Check if the active user exists
    if active_user_id not in USERS:
        return False
    
    # Check if the member user exists
    if member_user_id not in USERS:
        return False
    
    # Check if action_type is valid
    if action_type not in ['add', 'remove']:
        return False
    
    # Get user role and project details
    user_role = USERS[active_user_id]["role"]
    project = PROJECTS[prj_id]
    
    # Check if user has permission (admin or project owner)
    if user_role != "admin" and project["owner_id"] != active_user_id:
        return False
    
    # Perform the action
    if action_type == "add":
        if member_user_id not in project["members"]:
            project["members"].append(member_user_id)
        return True
    elif action_type == "remove":
        if member_user_id in project["members"]:
            project["members"].remove(member_user_id)
        return True
    
    return False
