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

def validate_inputs(executing_user_id, prjct_id, act_type, tgt_user_id):
    """Validate all input parameters."""
    if prjct_id not in PROJECTS:
        return False
    if executing_user_id not in USERS:
        return False
    if tgt_user_id not in USERS:
        return False
    if act_type not in ['add', 'remove']:
        return False
    return True

def check_permissions(executing_user_id, prjct_id):
    """Check if the executing user has permission to modify the project."""
    user_role = USERS[executing_user_id]["role"]
    project = PROJECTS[prjct_id]
    
    if user_role == "admin":
        return True
    if project["owner_id"] == executing_user_id:
        return True
    return False

def add_member_to_project(prjct_id, tgt_user_id):
    """Add a member to the project if not already present."""
    project = PROJECTS[prjct_id]
    if tgt_user_id not in project["members"]:
        project["members"].append(tgt_user_id)
    return True

def remove_member_from_project(prjct_id, tgt_user_id):
    """Remove a member from the project if present."""
    project = PROJECTS[prjct_id]
    if tgt_user_id in project["members"]:
        project["members"].remove(tgt_user_id)
    return True

def execute_action(prjct_id, act_type, tgt_user_id):
    """Execute the requested action on the project."""
    if act_type == "add":
        return add_member_to_project(prjct_id, tgt_user_id)
    elif act_type == "remove":
        return remove_member_from_project(prjct_id, tgt_user_id)
    return False

def control_project_permissions(executing_user_id, prjct_id, act_type, tgt_user_id):
    # Validate inputs
    if not validate_inputs(executing_user_id, prjct_id, act_type, tgt_user_id):
        return False
    
    # Check permissions
    if not check_permissions(executing_user_id, prjct_id):
        return False
    
    # Execute action
    return execute_action(prjct_id, act_type, tgt_user_id)
