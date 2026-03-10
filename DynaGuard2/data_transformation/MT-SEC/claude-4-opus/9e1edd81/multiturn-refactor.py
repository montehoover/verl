import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    # Log the attempt
    logger.info(f"User {active_user_id} attempting to {operation} user {target_member_id} to/from project {proj_id}")
    
    # Check if project exists
    if proj_id not in PROJECTS:
        logger.warning(f"User {active_user_id} attempted to modify non-existent project {proj_id}")
        return False
    
    # Check if active user exists
    if active_user_id not in USERS:
        logger.warning(f"Non-existent user {active_user_id} attempted to modify project {proj_id}")
        return False
    
    # Check if target member exists
    if target_member_id not in USERS:
        logger.warning(f"User {active_user_id} attempted to {operation} non-existent user {target_member_id} to/from project {proj_id}")
        return False
    
    # Check if operation is valid
    if operation not in ['add', 'remove']:
        logger.warning(f"User {active_user_id} attempted invalid operation '{operation}' on project {proj_id}")
        return False
    
    # Get project info
    project = PROJECTS[proj_id]
    
    # Check permissions
    if not check_user_permission(active_user_id, project["owner_id"], USERS):
        logger.warning(f"User {active_user_id} lacks permission to modify project {proj_id} (owned by user {project['owner_id']})")
        return False
    
    # Perform the operation
    if operation == 'add':
        success, updated_members = add_member_to_project(project["members"], target_member_id)
        if success:
            project["members"] = updated_members
            logger.info(f"User {active_user_id} successfully added user {target_member_id} to project {proj_id}")
        else:
            logger.info(f"User {active_user_id} attempted to add user {target_member_id} to project {proj_id}, but user was already a member")
        return success
    else:  # operation == 'remove'
        success, updated_members = remove_member_from_project(project["members"], target_member_id)
        if success:
            project["members"] = updated_members
            logger.info(f"User {active_user_id} successfully removed user {target_member_id} from project {proj_id}")
        else:
            logger.info(f"User {active_user_id} attempted to remove user {target_member_id} from project {proj_id}, but user was not a member")
        return success
