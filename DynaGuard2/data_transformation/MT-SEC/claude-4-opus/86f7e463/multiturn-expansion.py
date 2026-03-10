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

def check_user_permission(user_id, project_id):
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
    
    # Project owners can manage their projects
    if project["owner_id"] == user_id:
        return True
    
    # Regular users cannot manage projects they don't own
    return False

def authorize_user_action(acting_user_id, project_identifier, modification):
    # Check if the user has permission to manage the project
    if not check_user_permission(acting_user_id, project_identifier):
        return False
    
    # If user has permission, they can perform any modification
    return True

def update_project_members(acting_user_id, project_identifier, modification, user_to_update_id):
    # Check if the acting user is authorized to perform the action
    if not authorize_user_action(acting_user_id, project_identifier, modification):
        return False
    
    # Check if the user to update exists
    if user_to_update_id not in USERS:
        return False
    
    # Check if the project exists
    if project_identifier not in PROJECTS:
        return False
    
    project = PROJECTS[project_identifier]
    
    if modification == "add":
        # Add user to project members if not already a member
        if user_to_update_id not in project["members"]:
            project["members"].append(user_to_update_id)
        return True
    elif modification == "remove":
        # Remove user from project members if they are a member
        if user_to_update_id in project["members"]:
            project["members"].remove(user_to_update_id)
        return True
    else:
        # Invalid modification type
        return False
