import logging
from datetime import datetime

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


def check_project_exists(proj_id: str, projects_db: dict) -> dict:
    """Check if project exists and return it."""
    if proj_id not in projects_db:
        raise PermissionError(f"Project {proj_id} not found")
    return projects_db[proj_id]


def check_user_membership(owner_id: str, project: dict) -> None:
    """Check if user is a member of the project."""
    if owner_id not in project["members"]:
        raise PermissionError(f"User {owner_id} is not a member of project {project['project_id']}")


def check_access_permissions(owner_id: str, access_level: str, project: dict) -> None:
    """Check if user has permission to modify project settings based on access level."""
    if access_level == "viewer":
        raise PermissionError("Viewers do not have permission to modify project settings")
    elif access_level == "team_member":
        if owner_id != project["creator_id"]:
            raise PermissionError("Team members can only modify projects they created")
    elif access_level == "project_manager":
        # Project managers can modify any project they're a member of
        pass
    else:
        raise PermissionError(f"Invalid access level: {access_level}")


def update_project_settings(project: dict, modifications: dict) -> None:
    """Update project settings with modifications."""
    project["settings"].update(modifications)


def change_project_settings(owner_id: str, access_level: str, proj_id: str, modifications: dict) -> str:
    # Log the access attempt
    logger.info(f"User {owner_id} with access level '{access_level}' attempting to modify project {proj_id}")
    logger.info(f"Attempted modifications: {modifications}")
    
    try:
        # Check if project exists
        project = check_project_exists(proj_id, PROJECTS_DB)
        
        # Check if user is a member of the project
        check_user_membership(owner_id, project)
        
        # Check access level permissions
        check_access_permissions(owner_id, access_level, project)
        
        # Update the project settings
        update_project_settings(project, modifications)
        
        # Log successful update
        logger.info(f"User {owner_id} successfully updated project {proj_id} settings")
        logger.info(f"New settings: {project['settings']}")
        
        return f"Project {proj_id} settings have been successfully updated"
    
    except PermissionError as e:
        # Log failed access attempt
        logger.warning(f"Failed access attempt by user {owner_id} on project {proj_id}: {str(e)}")
        raise
