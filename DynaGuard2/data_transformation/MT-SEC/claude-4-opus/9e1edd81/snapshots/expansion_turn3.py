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

def get_user_role_and_project_owner(user_id, proj_id):
    user = USERS.get(user_id)
    project = PROJECTS.get(proj_id)
    
    user_role = user["role"] if user else None
    project_owner = project["owner_id"] if project else None
    
    return (user_role, project_owner)

def can_user_perform_action(active_user_id, proj_id, operation):
    user_role, project_owner = get_user_role_and_project_owner(active_user_id, proj_id)
    
    if user_role == "admin":
        return True
    
    if user_role == "user" and active_user_id == project_owner:
        return True
    
    return False

def handle_project_membership(active_user_id, proj_id, operation, target_member_id):
    if not can_user_perform_action(active_user_id, proj_id, operation):
        return False
    
    project = PROJECTS.get(proj_id)
    if not project:
        return False
    
    if operation == "add":
        if target_member_id not in project["members"]:
            project["members"].append(target_member_id)
            return True
        return False
    elif operation == "remove":
        if target_member_id in project["members"]:
            project["members"].remove(target_member_id)
            return True
        return False
    
    return False
