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
    # Check if user exists in USERS
    if user_id not in USERS:
        return "user"
    
    # Check if user is admin
    if USERS[user_id]["role"] == "admin":
        return "admin"
    
    # Check if user is a project owner
    for project_id, project_data in PROJECTS.items():
        if project_data["owner_id"] == user_id:
            return "owner"
    
    # Default to regular user
    return "user"

def can_modify_project(current_user_id, project_id):
    # Check if project exists
    if project_id not in PROJECTS:
        return False
    
    # Get user role
    user_role = check_user_role(current_user_id)
    
    # Admins can modify any project
    if user_role == "admin":
        return True
    
    # Check if user is the owner of this specific project
    if PROJECTS[project_id]["owner_id"] == current_user_id:
        return True
    
    # Regular users and owners of other projects cannot modify
    return False
