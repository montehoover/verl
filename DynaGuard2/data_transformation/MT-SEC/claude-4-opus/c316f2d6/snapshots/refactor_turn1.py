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

def manage_project_access(current_user_id: int, project_id: int, action: str, target_user_id: int) -> bool:
    # Check if the project exists
    if project_id not in PROJECTS:
        return False
    
    # Check if the current user exists
    if current_user_id not in USERS:
        return False
    
    # Check if the target user exists
    if target_user_id not in USERS:
        return False
    
    # Check if the action is valid
    if action not in ['add', 'remove']:
        return False
    
    # Get the current user's role
    current_user_role = USERS[current_user_id]["role"]
    
    # Get the project
    project = PROJECTS[project_id]
    
    # Check if the current user has permission to modify the project
    is_admin = current_user_role == "admin"
    is_owner = project["owner_id"] == current_user_id
    
    if not (is_admin or is_owner):
        return False
    
    # Perform the action
    if action == 'add':
        if target_user_id not in project["members"]:
            project["members"].append(target_user_id)
            return True
        else:
            return False
    elif action == 'remove':
        if target_user_id in project["members"]:
            project["members"].remove(target_user_id)
            return True
        else:
            return False
