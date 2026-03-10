import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

def change_project_config(uid: str, role: str, proj: str, settings_update: dict) -> str:
    logger.info(f"User {uid} ({role}) attempting to update settings for project {proj}: {settings_update}")
    
    # Check if project exists
    if proj not in PROJECTS_DB:
        logger.error(f"Failed attempt by {uid} ({role}): Project {proj} not found")
        raise PermissionError(f"Project {proj} not found")
    
    project = PROJECTS_DB[proj]
    
    # Viewers cannot modify project settings
    if role == 'viewer':
        logger.warning(f"Unauthorized attempt by {uid} ({role}): Viewers cannot modify project settings")
        raise PermissionError("Viewers are not authorized to modify project settings")
    
    # Unknown role
    if role not in ['project_manager', 'team_member']:
        logger.error(f"Failed attempt by {uid}: Unknown role {role}")
        raise PermissionError(f"Unknown role: {role}")
    
    # User must be a member of the project
    if uid not in project['members']:
        logger.warning(f"Unauthorized attempt by {uid} ({role}): Not a member of project {proj}")
        raise PermissionError(f"User {uid} is not a member of project {proj}")
    
    # Team members can only change settings if they are the creator
    if role == 'team_member' and uid != project['creator_id']:
        logger.warning(f"Unauthorized attempt by {uid} ({role}): Only project managers can modify settings for project {proj}")
        raise PermissionError("Only project managers can modify project settings")
    
    # Update the project settings
    old_settings = project['settings'].copy()
    project['settings'].update(settings_update)
    logger.info(f"Successfully updated settings for project {proj} by {uid} ({role}). Changed from {old_settings} to {project['settings']}")
    
    return f"Successfully updated settings for project {proj}"
