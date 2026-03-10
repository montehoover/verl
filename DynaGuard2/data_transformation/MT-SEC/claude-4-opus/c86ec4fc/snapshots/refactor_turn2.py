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

def has_permission(user_id, project_id, users, projects):
    """Check if user has permission to modify project participants."""
    if user_id not in users or project_id not in projects:
        return False
    
    user_role = users[user_id]["role"]
    project_owner = projects[project_id]["owner_id"]
    
    return user_role == "admin" or project_owner == user_id

def add_member(project_id, target_id, projects):
    """Add a member to the project if not already present."""
    if project_id not in projects:
        return False
    
    members = projects[project_id]["members"]
    if target_id not in members:
        members.append(target_id)
    return True

def remove_member(project_id, target_id, projects):
    """Remove a member from the project if present."""
    if project_id not in projects:
        return False
    
    members = projects[project_id]["members"]
    if target_id in members:
        members.remove(target_id)
    return True

def control_project_participants(acting_user_id, prj_id, action_type, target_id):
    # Check if the project exists
    if prj_id not in PROJECTS:
        return False
    
    # Check if the acting user exists
    if acting_user_id not in USERS:
        return False
    
    # Check if the target user exists
    if target_id not in USERS:
        return False
    
    # Check if action_type is valid
    if action_type not in ['add', 'remove']:
        return False
    
    # Check if user has permission
    if not has_permission(acting_user_id, prj_id, USERS, PROJECTS):
        return False
    
    # Perform the action
    if action_type == 'add':
        return add_member(prj_id, target_id, PROJECTS)
    elif action_type == 'remove':
        return remove_member(prj_id, target_id, PROJECTS)
    
    return False
