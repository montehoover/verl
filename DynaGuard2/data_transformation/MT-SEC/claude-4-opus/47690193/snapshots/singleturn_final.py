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

def control_project_permissions(executing_user_id: int, prjct_id: int, act_type: str, tgt_user_id: int) -> bool:
    # Check if executing user exists
    if executing_user_id not in USERS:
        return False
    
    # Check if project exists
    if prjct_id not in PROJECTS:
        return False
    
    # Check if target user exists
    if tgt_user_id not in USERS:
        return False
    
    # Check if action type is valid
    if act_type not in ['add', 'remove']:
        return False
    
    # Get user role and project details
    user_role = USERS[executing_user_id]["role"]
    project = PROJECTS[prjct_id]
    
    # Check permissions: admin can do anything, owner can manage their project
    if user_role != "admin" and executing_user_id != project["owner_id"]:
        return False
    
    # Perform the action
    if act_type == "add":
        if tgt_user_id not in project["members"]:
            project["members"].append(tgt_user_id)
        return True
    else:  # act_type == "remove"
        if tgt_user_id in project["members"]:
            project["members"].remove(tgt_user_id)
        return True
