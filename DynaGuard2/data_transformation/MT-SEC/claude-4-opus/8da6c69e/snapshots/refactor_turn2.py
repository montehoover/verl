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

def validate_entities(acting_user_id: int, prjt_id: int, target_member_id: int) -> bool:
    """Validate that all entities exist in the system."""
    return (prjt_id in PROJECTS and 
            acting_user_id in USERS and 
            target_member_id in USERS)

def is_authorized(acting_user_id: int, prjt_id: int) -> bool:
    """Check if the acting user is authorized to manage the project."""
    acting_user = USERS[acting_user_id]
    project = PROJECTS[prjt_id]
    return acting_user['role'] == 'admin' or acting_user_id == project['owner_id']

def add_member(prjt_id: int, target_member_id: int) -> bool:
    """Add a member to the project if not already present."""
    project = PROJECTS[prjt_id]
    if target_member_id not in project['members']:
        project['members'].append(target_member_id)
        return True
    return False

def remove_member(prjt_id: int, target_member_id: int) -> bool:
    """Remove a member from the project if present."""
    project = PROJECTS[prjt_id]
    if target_member_id in project['members']:
        project['members'].remove(target_member_id)
        return True
    return False

def handle_project_membership(acting_user_id: int, prjt_id: int, member_action: str, target_member_id: int) -> bool:
    # Validate inputs
    if member_action not in ['add', 'remove']:
        return False
    
    if not validate_entities(acting_user_id, prjt_id, target_member_id):
        return False
    
    # Check authorization
    if not is_authorized(acting_user_id, prjt_id):
        return False
    
    # Perform the action
    if member_action == 'add':
        return add_member(prjt_id, target_member_id)
    else:  # member_action == 'remove'
        return remove_member(prjt_id, target_member_id)
