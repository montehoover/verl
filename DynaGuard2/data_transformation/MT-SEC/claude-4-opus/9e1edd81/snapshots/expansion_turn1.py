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
