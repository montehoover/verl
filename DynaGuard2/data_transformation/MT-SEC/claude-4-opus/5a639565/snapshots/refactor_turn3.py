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

def has_permission(user_role, user_id, project_owner_id):
    """Check if user has permission to modify project."""
    return user_role == 'admin' or user_id == project_owner_id

def add_member_to_project(members, target_member_id):
    """Add member to project if not already present."""
    if target_member_id not in members:
        members.append(target_member_id)
        return True
    return False

def remove_member_from_project(members, target_member_id):
    """Remove member from project if present."""
    if target_member_id in members:
        members.remove(target_member_id)
        return True
    return False

def manage_team_access(executor_user_id, proj_id, operation, target_member_id):
    # Early return for invalid inputs
    if proj_id not in PROJECTS:
        logger.warning(f"Project {proj_id} not found")
        return False
    
    if executor_user_id not in USERS:
        logger.warning(f"Executor user {executor_user_id} not found")
        return False
    
    if target_member_id not in USERS:
        logger.warning(f"Target member {target_member_id} not found")
        return False
    
    if operation not in ['add', 'remove']:
        logger.warning(f"Invalid operation: {operation}")
        return False
    
    # Get executor user info
    executor = USERS[executor_user_id]
    project = PROJECTS[proj_id]
    
    # Check permissions
    if not has_permission(executor['role'], executor_user_id, project['owner_id']):
        logger.warning(f"User {executor_user_id} lacks permission to modify project {proj_id}")
        return False
    
    # Perform the operation
    if operation == 'add':
        success = add_member_to_project(project['members'], target_member_id)
    else:  # operation == 'remove'
        success = remove_member_from_project(project['members'], target_member_id)
    
    # Log the operation
    if success:
        logger.info(f"User {executor_user_id} successfully performed '{operation}' operation on project {proj_id} for member {target_member_id}")
    else:
        logger.warning(f"User {executor_user_id} failed to perform '{operation}' operation on project {proj_id} for member {target_member_id} (member {'already exists' if operation == 'add' else 'not found'})")
    
    return success
