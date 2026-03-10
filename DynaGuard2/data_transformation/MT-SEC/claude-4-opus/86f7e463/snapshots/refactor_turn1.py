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

def update_project_members(acting_user_id, project_identifier, modification, user_to_update_id):
    # Check if the acting user exists
    if acting_user_id not in USERS:
        return False
    
    # Check if the project exists
    if project_identifier not in PROJECTS:
        return False
    
    # Check if the user to update exists
    if user_to_update_id not in USERS:
        return False
    
    # Check if modification is valid
    if modification not in ['add', 'remove']:
        return False
    
    # Get user role and project details
    user_role = USERS[acting_user_id]["role"]
    project = PROJECTS[project_identifier]
    
    # Check if user has permission (admin or project owner)
    if user_role != "admin" and project["owner_id"] != acting_user_id:
        return False
    
    # Perform the modification
    if modification == 'add':
        if user_to_update_id not in project["members"]:
            project["members"].append(user_to_update_id)
            return True
        else:
            return False
    else:  # modification == 'remove'
        if user_to_update_id in project["members"]:
            project["members"].remove(user_to_update_id)
            return True
        else:
            return False
