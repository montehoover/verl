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

def has_project_access(user_id: int, project_id: int, users: dict, projects: dict) -> bool:
    """Check if a user has permission to modify a project."""
    if user_id not in users or project_id not in projects:
        return False
    
    user_role = users[user_id]["role"]
    project = projects[project_id]
    
    return user_role == "admin" or project["owner_id"] == user_id

def add_member_to_project(project: dict, user_id: int) -> bool:
    """Add a member to a project if they're not already a member."""
    if user_id not in project["members"]:
        project["members"].append(user_id)
        return True
    return False

def remove_member_from_project(project: dict, user_id: int) -> bool:
    """Remove a member from a project if they're currently a member."""
    if user_id in project["members"]:
        project["members"].remove(user_id)
        return True
    return False

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
    
    # Check permissions
    if not has_project_access(current_user_id, project_id, USERS, PROJECTS):
        return False
    
    # Get the project
    project = PROJECTS[project_id]
    
    # Perform the action
    if action == 'add':
        return add_member_to_project(project, target_user_id)
    elif action == 'remove':
        return remove_member_from_project(project, target_user_id)
