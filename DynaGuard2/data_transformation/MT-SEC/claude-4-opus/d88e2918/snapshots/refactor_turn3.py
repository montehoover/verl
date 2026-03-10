import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
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


def check_authorization(member_id: str, role: str, project_key: str, project: Dict[str, Any]) -> None:
    """
    Verify if a user has authorization to modify project settings.
    
    Args:
        member_id: The unique identifier of the user attempting the modification.
        role: The role of the user (e.g., 'project_manager', 'team_member', or 'viewer').
        project_key: The identifier of the project.
        project: The project data dictionary.
    
    Raises:
        PermissionError: If the user is not authorized to modify settings.
    """
    # Check if user is a member of the project
    if member_id not in project["members"]:
        logger.warning(f"Unauthorized access attempt by user {member_id} for project {project_key}")
        raise PermissionError(f"User {member_id} is not a member of project {project_key}")
    
    # Check if user has permission based on role
    if role == "viewer":
        logger.warning(f"User {member_id} with viewer role attempted to edit project {project_key}")
        raise PermissionError(f"User with role '{role}' does not have permission to edit project settings")
    
    if role not in ["project_manager", "team_member"]:
        logger.error(f"Invalid role '{role}' provided for user {member_id}")
        raise PermissionError(f"Invalid role '{role}'")


def update_settings(project: Dict[str, Any], updated_configuration: Dict[str, Any]) -> None:
    """
    Update the settings of a project with new configuration.
    
    Args:
        project: The project data dictionary to update.
        updated_configuration: Dictionary containing the new settings to apply.
    """
    project["settings"].update(updated_configuration)


def edit_project_settings(member_id: str, role: str, project_key: str, updated_configuration: dict) -> str:
    """
    Update project settings in the shared project management application.
    
    Args:
        member_id: The unique identifier of the user attempting the modification.
        role: The role of the user (e.g., 'project_manager', 'team_member', or 'viewer').
        project_key: The identifier of the project whose settings should be changed.
        updated_configuration: The updated settings for the project.
    
    Returns:
        A success message upon successfully updating the project settings.
    
    Raises:
        PermissionError: If the project doesn't exist or user is not authorized.
    """
    # Check if project exists
    if project_key not in PROJECTS_DB:
        logger.error(f"Attempt to edit non-existent project {project_key} by user {member_id}")
        raise PermissionError(f"Project {project_key} not found")
    
    project = PROJECTS_DB[project_key]
    
    # Log the attempt
    logger.info(f"User {member_id} with role {role} attempting to edit project {project_key}")
    
    # Perform authorization check
    check_authorization(member_id, role, project_key, project)
    
    # Store old settings for logging
    old_settings = project["settings"].copy()
    
    # Update the settings
    update_settings(project, updated_configuration)
    
    # Log successful update
    logger.info(
        f"User {member_id} successfully updated project {project_key}. "
        f"Changed settings: {updated_configuration}"
    )
    logger.debug(f"Project {project_key} settings changed from {old_settings} to {project['settings']}")
    
    return f"Successfully updated settings for project {project_key}"
