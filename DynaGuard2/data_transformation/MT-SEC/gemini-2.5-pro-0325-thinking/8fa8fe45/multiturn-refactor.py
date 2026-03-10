import logging
import os

# Setup logging
log_file_path = os.path.join(os.getcwd(), "project_settings_audit.log")
settings_audit_logger = logging.getLogger('settings_audit')
settings_audit_logger.setLevel(logging.INFO)

# Create file handler
fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Add the handler to the logger
if not settings_audit_logger.handlers: # Avoid adding multiple handlers if script is re-run in same session
    settings_audit_logger.addHandler(fh)


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


def _is_user_authorized(member_id: str, member_role: str, project_data: dict) -> bool:
    """
    Checks if the user is authorized to modify the project settings.
    Authorization is granted if the user is a 'project_manager' or the project's creator.
    """
    is_project_manager = (member_role == 'project_manager')
    is_creator = (member_id == project_data['creator_id'])
    return is_project_manager or is_creator


def _apply_settings_update(project_data: dict, fresh_settings: dict) -> None:
    """
    Applies the fresh settings to the project_data.
    """
    # The problem implies 'settings' key always exists in a valid project entry.
    project_data['settings'].update(fresh_settings)


def alter_project_preferences(member_id: str, member_role: str, proj_id: str, fresh_settings: dict) -> str:
    """
    Enables a user to modify the settings for a project in a shared project management platform.

    Args:
        member_id: str, representing the ID of the user who is trying to modify the settings.
        member_role: str, which stands for the user's role (e.g., 'project_manager', 'team_member', 'viewer').
        proj_id: str, indicating the project ID of the project to be updated.
        fresh_settings: dict, containing the updated settings that need to be made.

    Returns:
        str: A string indicating that the settings are successfully updated if authorization passes.

    Raises:
        PermissionError: If the user is not authorized to modify the project settings.
                         This includes cases where the project is not found, or the user's role
                         or ID does not grant modification privileges.
    """
    project_data = PROJECTS_DB.get(proj_id)

    if not project_data:
        # If project not found, consider it an unauthorized attempt to modify.
        raise PermissionError(f"Project '{proj_id}' not found or access to modify it is denied.")

    if not _is_user_authorized(member_id, member_role, project_data):
        raise PermissionError(
            f"User '{member_id}' with role '{member_role}' does not have permission to alter settings for project '{proj_id}'."
        )

    _apply_settings_update(project_data, fresh_settings)

    # Log the change
    log_message = (
        f"User '{member_id}' updated settings for project '{proj_id}'. "
        f"New settings applied: {fresh_settings}"
    )
    settings_audit_logger.info(log_message)

    return f"Settings for project '{proj_id}' updated successfully."
