import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='project_audit.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

def check_user_permissions(user_token: str, role: str, project: dict) -> bool:
    """Check if user has permission to modify the project."""
    if role == 'viewer':
        return False
    
    if role in ['project_manager', 'team_member']:
        # Check if user is either the creator or a member of the project
        return user_token == project['creator_id'] or user_token in project['members']
    
    # Invalid role
    return False

def update_project_settings(project: dict, updated_values: dict) -> None:
    """Update the project settings with new values."""
    project['settings'].update(updated_values)

def adjust_project_params(user_token: str, role: str, project_ident: str, updated_values: dict) -> str:
    # Log the attempt
    logging.info(f"User '{user_token}' with role '{role}' attempting to update project '{project_ident}' with values: {updated_values}")
    
    # Check if project exists
    if project_ident not in PROJECTS_DB:
        logging.error(f"Failed update attempt by user '{user_token}' - Project '{project_ident}' not found")
        raise PermissionError("Project not found")
    
    project = PROJECTS_DB[project_ident]
    
    # Check permissions
    if not check_user_permissions(user_token, role, project):
        if role == 'viewer':
            logging.warning(f"Authorization failed for user '{user_token}' - Viewers cannot update project '{project_ident}'")
            raise PermissionError("Viewers do not have permission to update project settings")
        elif role not in ['project_manager', 'team_member']:
            logging.warning(f"Authorization failed for user '{user_token}' - Invalid role '{role}' for project '{project_ident}'")
            raise PermissionError("Invalid user role")
        else:
            logging.warning(f"Authorization failed for user '{user_token}' - Not authorized to update project '{project_ident}'")
            raise PermissionError("User is not authorized to update this project")
    
    # Update settings
    old_settings = project['settings'].copy()
    update_project_settings(project, updated_values)
    
    # Log successful update
    logging.info(f"User '{user_token}' successfully updated project '{project_ident}' - Old settings: {old_settings}, New settings: {project['settings']}")
    
    return f"Project {project_ident} settings have been successfully updated"
