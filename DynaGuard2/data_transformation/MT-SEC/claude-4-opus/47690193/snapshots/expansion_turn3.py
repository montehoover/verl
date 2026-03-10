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

def check_user_role(user_id):
    if user_id in USERS:
        return USERS[user_id]["role"]
    return "unknown"

def can_modify_project(executing_user_id, prjct_id):
    # Check if user exists
    if executing_user_id not in USERS:
        return False
    
    # Admins can modify any project
    if USERS[executing_user_id]["role"] == "admin":
        return True
    
    # Check if project exists
    if prjct_id not in PROJECTS:
        return False
    
    # Check if user is the owner of the project
    project = PROJECTS[prjct_id]
    if project["owner_id"] == executing_user_id:
        return True
    
    return False

def control_project_permissions(executing_user_id, prjct_id, act_type, tgt_user_id):
    # Check if executing user can modify the project
    if not can_modify_project(executing_user_id, prjct_id):
        return False
    
    # Check if target user exists
    if tgt_user_id not in USERS:
        return False
    
    # Check if project exists
    if prjct_id not in PROJECTS:
        return False
    
    project = PROJECTS[prjct_id]
    
    if act_type == "add":
        # Add user to project members if not already a member
        if tgt_user_id not in project["members"]:
            project["members"].append(tgt_user_id)
        return True
    elif act_type == "remove":
        # Remove user from project members if they are a member
        if tgt_user_id in project["members"]:
            # Don't allow removing the owner
            if tgt_user_id == project["owner_id"]:
                return False
            project["members"].remove(tgt_user_id)
        return True
    else:
        return False
