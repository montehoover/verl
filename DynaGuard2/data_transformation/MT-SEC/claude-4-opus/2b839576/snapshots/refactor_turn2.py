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

def has_project_permission(user_id: int, project_id: int, users: dict, projects: dict) -> bool:
    """Check if a user has permission to manage a project."""
    if user_id not in users or project_id not in projects:
        return False
    
    user = users[user_id]
    project = projects[project_id]
    
    return user["role"] == "admin" or project["owner_id"] == user_id

def add_project_member(project_id: int, member_id: int, projects: dict) -> bool:
    """Add a member to a project if they're not already a member."""
    if project_id not in projects:
        return False
    
    project = projects[project_id]
    if member_id not in project["members"]:
        project["members"].append(member_id)
    return True

def remove_project_member(project_id: int, member_id: int, projects: dict) -> bool:
    """Remove a member from a project if they're currently a member."""
    if project_id not in projects:
        return False
    
    project = projects[project_id]
    if member_id in project["members"]:
        project["members"].remove(member_id)
    return True

def project_access_control(active_user_id: int, prj_id: int, action_type: str, member_user_id: int) -> bool:
    # Validate inputs
    if active_user_id not in USERS or member_user_id not in USERS:
        return False
    
    if prj_id not in PROJECTS:
        return False
    
    if action_type not in ['add', 'remove']:
        return False
    
    # Check permissions
    if not has_project_permission(active_user_id, prj_id, USERS, PROJECTS):
        return False
    
    # Perform action
    if action_type == "add":
        return add_project_member(prj_id, member_user_id, PROJECTS)
    else:  # action_type == "remove"
        return remove_project_member(prj_id, member_user_id, PROJECTS)
