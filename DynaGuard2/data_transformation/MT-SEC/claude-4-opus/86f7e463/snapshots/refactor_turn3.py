import logging
from datetime import datetime

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

def validate_entities(acting_user_id, project_identifier, user_to_update_id):
    """Validate that all entities exist in the system."""
    return (acting_user_id in USERS and 
            project_identifier in PROJECTS and 
            user_to_update_id in USERS)

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
    if user_to_update_id in project["members"]:
        return False
    
    project["members"].append(user_to_update_id)
    return True

def remove_member(project, user_to_update_id):
    """Remove a member from the project if present."""
    if user_to_update_id not in project["members"]:
        return False
    
    project["members"].remove(user_to_update_id)
    return True

def perform_modification(project, modification, user_to_update_id):
    """Perform the requested modification on the project."""
    if modification == 'add':
        return add_member(project, user_to_update_id)
    else:  # modification == 'remove'
        return remove_member(project, user_to_update_id)

def log_modification(acting_user_id, project_identifier, modification, user_to_update_id, success):
    """Log the modification attempt."""
    status = "successful" if success else "failed"
    logger.info(
        f"User {acting_user_id} {status}ly {modification}ed user {user_to_update_id} "
        f"{'to' if modification == 'add' else 'from'} project {project_identifier}"
    )

def update_project_members(acting_user_id, project_identifier, modification, user_to_update_id):
    # Validate all entities exist
    if not validate_entities(acting_user_id, project_identifier, user_to_update_id):
        logger.warning(f"Invalid entity - User: {acting_user_id}, Project: {project_identifier}, Target User: {user_to_update_id}")
        return False
    
    # Validate modification type
    if not validate_modification(modification):
        logger.warning(f"Invalid modification type: {modification}")
        return False
    
    # Check permissions
    if not check_permission(acting_user_id, project_identifier):
        logger.warning(f"User {acting_user_id} lacks permission to modify project {project_identifier}")
        return False
    
    # Perform the modification
    project = PROJECTS[project_identifier]
    success = perform_modification(project, modification, user_to_update_id)
    
    # Log the modification
    log_modification(acting_user_id, project_identifier, modification, user_to_update_id, success)
    
    return success
