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

def has_permission(user_id, proj_id):
    # Check if user exists
    if user_id not in USERS:
        return False
    
    # Check if project exists
    if proj_id not in PROJECTS:
        return False
    
    user = USERS[user_id]
    project = PROJECTS[proj_id]
    
    # Admins can manage any project
    if user["role"] == "admin":
        return True
    
    # Project owners can manage their own projects
    if project["owner_id"] == user_id:
        return True
    
    # Regular users cannot manage projects they don't own
    return False

def can_perform_action(acting_user_id, prj_id, action_type):
    # Check if user has basic permission to manage the project
    if not has_permission(acting_user_id, prj_id):
        return False
    
    # Define allowed actions
    allowed_actions = ["add_member", "remove_member", "update_project", "delete_project"]
    
    # Check if the action type is valid
    if action_type not in allowed_actions:
        return False
    
    # If user has permission and action is valid, allow the action
    return True
