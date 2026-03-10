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

def has_project_permission(user_id: int, project_id: int, users: dict, projects: dict) -> bool:
    """Check if a user has permission to manage a project."""
    if user_id not in users:
        logger.error(f"User {user_id} not found")
        return False
    
    if project_id not in projects:
        logger.error(f"Project {project_id} not found")
        return False
    
    user = users[user_id]
    project = projects[project_id]
    
    is_admin = user["role"] == "admin"
    is_owner = project["owner_id"] == user_id
    
    return is_admin or is_owner

def add_project_member(project_id: int, member_id: int, projects: dict) -> bool:
    """Add a member to a project if they're not already a member."""
    if project_id not in projects:
        logger.error(f"Cannot add member to non-existent project {project_id}")
        return False
    
    project = projects[project_id]
    if member_id in project["members"]:
        logger.info(f"User {member_id} is already a member of project {project_id}")
        return True
    
    project["members"].append(member_id)
    logger.info(f"Added user {member_id} to project {project_id}")
    return True

def remove_project_member(project_id: int, member_id: int, projects: dict) -> bool:
    """Remove a member from a project if they're currently a member."""
    if project_id not in projects:
        logger.error(f"Cannot remove member from non-existent project {project_id}")
        return False
    
    project = projects[project_id]
    if member_id not in project["members"]:
        logger.info(f"User {member_id} is not a member of project {project_id}")
        return True
    
    project["members"].remove(member_id)
    logger.info(f"Removed user {member_id} from project {project_id}")
    return True

def project_access_control(active_user_id: int, prj_id: int, action_type: str, member_user_id: int) -> bool:
    # Validate user existence
    if active_user_id not in USERS:
        logger.error(f"Active user {active_user_id} not found")
        return False
    
    if member_user_id not in USERS:
        logger.error(f"Member user {member_user_id} not found")
        return False
    
    # Validate project existence
    if prj_id not in PROJECTS:
        logger.error(f"Project {prj_id} not found")
        return False
    
    # Validate action type
    if action_type not in ['add', 'remove']:
        logger.error(f"Invalid action type: {action_type}")
        return False
    
    # Check permissions
    if not has_project_permission(active_user_id, prj_id, USERS, PROJECTS):
        logger.warning(f"User {active_user_id} lacks permission to manage project {prj_id}")
        return False
    
    # Log the action attempt
    logger.info(f"User {active_user_id} attempting to {action_type} user {member_user_id} to/from project {prj_id}")
    
    # Perform action
    if action_type == "add":
        return add_project_member(prj_id, member_user_id, PROJECTS)
    else:  # action_type == "remove"
        return remove_project_member(prj_id, member_user_id, PROJECTS)
