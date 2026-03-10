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


def _has_permission_to_update(uid: str, role: str, project: dict) -> bool:
    """Checks if the user has permission to update the project settings."""
    is_project_manager = role == 'project_manager'
    is_creator = project["creator_id"] == uid
    return is_project_manager or is_creator


def _apply_settings_update(project: dict, settings: dict) -> None:
    """Applies the settings update to the project."""
    project["settings"].update(settings)


def update_project_configurations(uid: str, role: str, pid: str, settings: dict) -> str:
    """
    Allows authorized users to update project settings.

    Args:
        uid: The identifier representing the user attempting to update settings.
        role: The user's role (e.g., 'project_manager', 'team_member', 'viewer').
        pid: The identifier of the project being modified.
        settings: A dictionary containing the new configuration for the project.

    Returns:
        A success message indicating that the settings have been updated.

    Raises:
        PermissionError: If the user does not have sufficient rights.
        ValueError: If the project ID is not found.
    """
    if pid not in PROJECTS_DB:
        raise ValueError(f"Project with ID '{pid}' not found.")

    project = PROJECTS_DB[pid]

    if not _has_permission_to_update(uid, role, project):
        error_message = f"User '{uid}' with role '{role}' attempted to update settings for project '{pid}' without authorization."
        logger.warning(error_message)
        raise PermissionError(error_message)

    _apply_settings_update(project, settings)
    success_message = f"Settings for project '{pid}' updated successfully by user '{uid}'."
    logger.info(success_message)
    return success_message
