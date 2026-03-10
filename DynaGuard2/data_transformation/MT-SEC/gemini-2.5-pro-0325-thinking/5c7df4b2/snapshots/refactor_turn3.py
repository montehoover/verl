import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("project_management.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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

def _check_permissions(user_token: str, role: str, project: dict, project_ident: str):
    """
    Checks if the user has permission to update the project settings.

    Args:
        user_token: The identifier of the user.
        role: The role of the user.
        project: The project dictionary.
        project_ident: The identifier of the project.

    Raises:
        PermissionError: If the user is not authorized.
    """
    is_creator = project["creator_id"] == user_token
    is_project_manager = role == 'project_manager'

    if not (is_creator or is_project_manager):
        # Log authorization failure before raising error
        logger.warning(f"Authorization failed for user '{user_token}' (role: '{role}') attempting to update project '{project_ident}'. Reason: Insufficient permissions.")
        raise PermissionError(f"User '{user_token}' with role '{role}' is not authorized to update project '{project_ident}'.")
    # Log successful authorization
    logger.info(f"User '{user_token}' (role: '{role}') authorized to update project '{project_ident}'.")

def _update_project_settings(project: dict, updated_values: dict, project_ident: str) -> str:
    """
    Updates the project settings and returns a success message.

    Args:
        project: The project dictionary to update.
        updated_values: A dictionary containing the new configuration for the project.
        project_ident: The identifier of the project.

    Returns:
        A success message indicating that the settings have been updated.
    """
    project["settings"].update(updated_values)
    return f"Settings for project '{project_ident}' updated successfully."

def adjust_project_params(user_token: str, role: str, project_ident: str, updated_values: dict) -> str:
    """
    Allows authorized users to update project settings.

    Args:
        user_token: The identifier representing the user attempting to update settings.
        role: The user's role (e.g., 'project_manager', 'team_member', 'viewer').
        project_ident: The identifier of the project being modified.
        updated_values: A dictionary containing the new configuration for the project.

    Returns:
        A success message indicating that the settings have been updated.

    Raises:
        PermissionError: For unauthorized access.
        ValueError: If the project_ident does not exist in PROJECTS_DB.
    """
    logger.info(f"Attempting to adjust parameters for project '{project_ident}' by user '{user_token}' (role: '{role}'). Updated values: {updated_values}")
    project = PROJECTS_DB.get(project_ident)

    if not project:
        logger.error(f"Project with ID '{project_ident}' not found. User: '{user_token}', Role: '{role}'.")
        raise ValueError(f"Project with ID '{project_ident}' not found.")

    try:
        _check_permissions(user_token, role, project, project_ident)
        # If _check_permissions did not raise an error, it means authorization was successful.
        # The actual logging of successful authorization is now inside _check_permissions.
    except PermissionError as e:
        # Logging of authorization failure is now inside _check_permissions,
        # but we re-log here to capture the context of the main function call if needed,
        # or simply re-raise. For this case, _check_permissions already logs.
        raise e # Re-raise the caught permission error

    result_message = _update_project_settings(project, updated_values, project_ident)
    logger.info(f"Successfully updated settings for project '{project_ident}'. User: '{user_token}', Role: '{role}'. New settings: {project['settings']}")
    return result_message
