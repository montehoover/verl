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


def can_perform_action(acting_user_id, prjt_id, member_action):
    # Check if the acting user has permission to manage the project
    if not has_permission_to_manage(acting_user_id, prjt_id):
        return False
    
    # Validate action type
    if member_action not in ["add", "remove"]:
        return False
    
    return True


def handle_project_membership(acting_user_id, prjt_id, member_action, target_member_id):
    # Check if the action can be performed
    if not can_perform_action(acting_user_id, prjt_id, member_action):
        return False
    
    # Check if target member exists
    if target_member_id not in USERS:
        return False
    
    # Check if project exists
    if prjt_id not in PROJECTS:
        return False
    
    project = PROJECTS[prjt_id]
    
    if member_action == "add":
        # Check if member is already in the project
        if target_member_id in project["members"]:
            return False
        # Add the member
        project["members"].append(target_member_id)
        return True
    elif member_action == "remove":
        # Check if member is in the project
        if target_member_id not in project["members"]:
            return False
        # Don't allow removing the owner
        if target_member_id == project["owner_id"]:
            return False
        # Remove the member
        project["members"].remove(target_member_id)
        return True
    
    return False
