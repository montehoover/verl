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


def has_permission_to_manage(user_id, project_id):
    # Check if user exists
    if user_id not in USERS:
        return False
    
    # Check if project exists
    if project_id not in PROJECTS:
        return False
    
    user = USERS[user_id]
    project = PROJECTS[project_id]
    
    # Admins can manage any project
    if user["role"] == "admin":
        return True
    
    # Project owners can manage their own projects
    if project["owner_id"] == user_id:
        return True
    
    return False
