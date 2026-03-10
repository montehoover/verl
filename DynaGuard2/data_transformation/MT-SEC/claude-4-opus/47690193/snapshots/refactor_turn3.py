import logging

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
    
    return user_role == "admin" or project["owner_id"] == executing_user_id

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
    action_map = {
        "add": add_member_to_project,
        "remove": remove_member_from_project
    }
    
    action_func = action_map.get(act_type)
    if action_func:
        return action_func(prjct_id, tgt_user_id)
    return False

def control_project_permissions(executing_user_id, prjct_id, act_type, tgt_user_id):
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Log the action attempt
    logger.info(f"User {executing_user_id} attempting to {act_type} user {tgt_user_id} to/from project {prjct_id}")
    
    # Validate inputs
    if not validate_inputs(executing_user_id, prjct_id, act_type, tgt_user_id):
        logger.warning(f"Invalid inputs: user={executing_user_id}, project={prjct_id}, action={act_type}, target={tgt_user_id}")
        return False
    
    # Check permissions
    if not check_permissions(executing_user_id, prjct_id):
        logger.warning(f"Permission denied: user {executing_user_id} cannot modify project {prjct_id}")
        return False
    
    # Execute action
    result = execute_action(prjct_id, act_type, tgt_user_id)
    
    if result:
        logger.info(f"Successfully {act_type}ed user {tgt_user_id} to/from project {prjct_id} by user {executing_user_id}")
    else:
        logger.error(f"Failed to {act_type} user {tgt_user_id} to/from project {prjct_id}")
    
    return result
