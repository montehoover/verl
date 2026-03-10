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
        logger.warning(f"Invalid action '{member_action}' attempted by user {acting_user_id} on project {prjt_id}")
        return False
    
    if not validate_entities(acting_user_id, prjt_id, target_member_id):
        logger.warning(f"Invalid entities - user {acting_user_id} attempted {member_action} on project {prjt_id} for member {target_member_id}")
        return False
    
    # Check authorization
    if not is_authorized(acting_user_id, prjt_id):
        logger.warning(f"Unauthorized access - user {acting_user_id} attempted {member_action} on project {prjt_id}")
        return False
    
    # Perform the action
    if member_action == 'add':
        success = add_member(prjt_id, target_member_id)
        if success:
            logger.info(f"User {acting_user_id} successfully added member {target_member_id} to project {prjt_id}")
        else:
            logger.info(f"User {acting_user_id} attempted to add member {target_member_id} to project {prjt_id} but member already exists")
    else:  # member_action == 'remove'
        success = remove_member(prjt_id, target_member_id)
        if success:
            logger.info(f"User {acting_user_id} successfully removed member {target_member_id} from project {prjt_id}")
        else:
            logger.info(f"User {acting_user_id} attempted to remove member {target_member_id} from project {prjt_id} but member not found")
    
    return success
