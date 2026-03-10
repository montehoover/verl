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

def validate_entities(acting_user_id, project_identifier, user_to_update_id):
    """Validate that all entities exist in the system."""
    if acting_user_id not in USERS:
        return False
    if project_identifier not in PROJECTS:
        return False
    if user_to_update_id not in USERS:
        return False
    return True

def validate_modification(modification):
    """Validate that the modification type is valid."""
    return modification in ['add', 'remove']

def check_permission(acting_user_id, project_identifier):
    """Check if the acting user has permission to modify the project."""
    user_role = USERS[acting_user_id]["role"]
    project = PROJECTS[project_identifier]
    return user_role == "admin" or project["owner_id"] == acting_user_id

def add_member(project, user_to_update_id):
    """Add a member to the project if not already present."""
    if user_to_update_id not in project["members"]:
        project["members"].append(user_to_update_id)
        return True
    return False

def remove_member(project, user_to_update_id):
    """Remove a member from the project if present."""
    if user_to_update_id in project["members"]:
        project["members"].remove(user_to_update_id)
        return True
    return False

def perform_modification(project, modification, user_to_update_id):
    """Perform the requested modification on the project."""
    if modification == 'add':
        return add_member(project, user_to_update_id)
    else:  # modification == 'remove'
        return remove_member(project, user_to_update_id)

def update_project_members(acting_user_id, project_identifier, modification, user_to_update_id):
    # Validate all entities exist
    if not validate_entities(acting_user_id, project_identifier, user_to_update_id):
        return False
    
    # Validate modification type
    if not validate_modification(modification):
        return False
    
    # Check permissions
    if not check_permission(acting_user_id, project_identifier):
        return False
    
    # Perform the modification
    project = PROJECTS[project_identifier]
    return perform_modification(project, modification, user_to_update_id)
