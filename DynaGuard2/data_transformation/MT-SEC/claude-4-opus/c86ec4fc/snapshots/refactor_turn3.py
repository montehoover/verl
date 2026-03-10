import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    # Validate all inputs first
    validation_errors = []
    if prj_id not in PROJECTS:
        validation_errors.append("project")
    if acting_user_id not in USERS:
        validation_errors.append("acting user")
    if target_id not in USERS:
        validation_errors.append("target user")
    if action_type not in ['add', 'remove']:
        validation_errors.append("action type")
    
    if validation_errors:
        logger.warning(f"Invalid {', '.join(validation_errors)} - Acting user: {acting_user_id}, Project: {prj_id}, Action: {action_type}, Target: {target_id}")
        return False
    
    # Check permission
    if not has_permission(acting_user_id, prj_id, USERS, PROJECTS):
        logger.warning(f"Permission denied - Acting user: {acting_user_id}, Project: {prj_id}, Action: {action_type}, Target: {target_id}")
        return False
    
    # Perform the action
    action_functions = {
        'add': add_member,
        'remove': remove_member
    }
    
    result = action_functions[action_type](prj_id, target_id, PROJECTS)
    
    if result:
        logger.info(f"Action successful - Acting user: {acting_user_id}, Project: {prj_id}, Action: {action_type}, Target: {target_id}")
    else:
        logger.error(f"Action failed - Acting user: {acting_user_id}, Project: {prj_id}, Action: {action_type}, Target: {target_id}")
    
    return result
