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

def check_user_permission(user_id: int, project_owner_id: int, users_data: dict) -> bool:
    """Check if user has permission to modify project membership."""
    if user_id not in users_data:
        return False
    
    user_role = users_data[user_id]["role"]
    return user_role == "admin" or project_owner_id == user_id

def add_member_to_project(members: list, member_id: int) -> tuple[bool, list]:
    """Add member to project. Returns (success, updated_members)."""
    if member_id in members:
        return False, members
    
    updated_members = members.copy()
    updated_members.append(member_id)
    return True, updated_members

def remove_member_from_project(members: list, member_id: int) -> tuple[bool, list]:
    """Remove member from project. Returns (success, updated_members)."""
    if member_id not in members:
        return False, members
    
    updated_members = members.copy()
    updated_members.remove(member_id)
    return True, updated_members

def handle_project_membership(active_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    # Check if project exists
    if proj_id not in PROJECTS:
        return False
    
    # Check if active user exists
    if active_user_id not in USERS:
        return False
    
    # Check if target member exists
    if target_member_id not in USERS:
        return False
    
    # Check if operation is valid
    if operation not in ['add', 'remove']:
        return False
    
    # Get project info
    project = PROJECTS[proj_id]
    
    # Check permissions
    if not check_user_permission(active_user_id, project["owner_id"], USERS):
        return False
    
    # Perform the operation
    if operation == 'add':
        success, updated_members = add_member_to_project(project["members"], target_member_id)
        if success:
            project["members"] = updated_members
        return success
    else:  # operation == 'remove'
        success, updated_members = remove_member_from_project(project["members"], target_member_id)
        if success:
            project["members"] = updated_members
        return success
