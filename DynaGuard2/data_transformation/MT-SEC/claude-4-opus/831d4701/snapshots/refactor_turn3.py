import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECTS_DB = {
    "PROJ001": {
        "project_id": "PROJ001",
        "creator_id": "USER1",
        "members": ["USER1", "USER2", "USER3"],
        "settings": {"visibility": "private", "deadline": "2023-12-31"}
    },
    "PROJ002": {
        "project_id": "PROJ002",
        "creator_id": "USER4",
        "members": ["USER4", "USER5"],
        "settings": {"visibility": "public", "deadline": "2023-10-15"}
    }
}

def check_project_exists(pid: str, projects_db: dict) -> bool:
    """Check if a project exists in the database."""
    return pid in projects_db

def is_user_authorized(uid: str, role: str, project: dict) -> bool:
    """Check if user has permission to update project settings."""
    return role == 'project_manager' or uid == project['creator_id']

def is_user_member(uid: str, project: dict) -> bool:
    """Check if user is a member of the project."""
    return uid in project['members']

def update_settings(project: dict, new_settings: dict) -> None:
    """Update project settings with new values."""
    project['settings'].update(new_settings)

def update_project_configurations(uid: str, role: str, pid: str, settings: dict) -> str:
    logger.info(f"User {uid} (role: {role}) attempting to update settings for project {pid}")
    
    # Check if project exists
    if not check_project_exists(pid, PROJECTS_DB):
        logger.warning(f"Unauthorized access attempt: User {uid} tried to update non-existent project {pid}")
        raise PermissionError(f"Project {pid} not found")
    
    project = PROJECTS_DB[pid]
    
    # Check if user is authorized
    if not is_user_authorized(uid, role, project):
        logger.warning(f"Unauthorized access attempt: User {uid} (role: {role}) lacks permission to update project {pid}")
        raise PermissionError(f"User {uid} with role {role} is not authorized to update project {pid} settings")
    
    # Check if user is a member of the project
    if not is_user_member(uid, project):
        logger.warning(f"Unauthorized access attempt: User {uid} is not a member of project {pid}")
        raise PermissionError(f"User {uid} is not a member of project {pid}")
    
    # Log the settings being changed
    old_settings = project['settings'].copy()
    logger.info(f"Updating project {pid} settings. Old settings: {old_settings}, New settings: {settings}")
    
    # Update the settings
    update_settings(project, settings)
    
    logger.info(f"Successfully updated project {pid} settings by user {uid} (role: {role})")
    
    return f"Project {pid} settings successfully updated"
