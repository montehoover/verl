import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def _is_user_authorized(member_id: str, role: str, project: dict) -> bool:
    """
    Checks if a user is authorized to edit project settings.

    A user is considered authorized if their role is 'project_manager' or
    if their member_id matches the creator_id of the project.

    Args:
        member_id: The unique identifier of the user.
        role: The role of the user (e.g., 'project_manager').
        project: The project dictionary containing 'creator_id'.

    Returns:
        True if the user is authorized, False otherwise.
    """
    return (role == 'project_manager') or (member_id == project['creator_id'])


def _update_project_settings_in_db(project_key: str, project: dict, updated_configuration: dict) -> None:
    """
    Updates the settings of a given project dictionary in PROJECTS_DB.

    This function directly modifies the 'settings' field of the project
    dictionary and logs the update.

    Args:
        project_key: The key of the project being updated (for logging).
        project: The project dictionary to update.
        updated_configuration: A dictionary containing the settings to be updated.
    """
    project['settings'].update(updated_configuration)
    logging.info(f"Project settings for '{project_key}' updated with: {updated_configuration}")


def edit_project_settings(member_id: str, role: str, project_key: str, updated_configuration: dict) -> str:
    """
    Updates project settings in a shared project management application.

    This function orchestrates the process of checking user authorization
    and then applying the requested settings changes to a specific project.

    Args:
        member_id: The unique identifier of the user attempting the modification.
        role: The role of the user (e.g., 'project_manager', 'team_member', 'viewer').
        project_key: The identifier of the project whose settings should be changed.
        updated_configuration: The updated settings for the project.

    Returns:
        A success message upon successfully updating the project settings.

    Raises:
        ValueError: If the project_key does not exist in PROJECTS_DB.
        PermissionError: For an unauthorized access.
    """
    if project_key not in PROJECTS_DB:
        logging.error(f"Attempt to edit non-existent project '{project_key}' by user '{member_id}'.")
        raise ValueError(f"Project with key '{project_key}' not found.")

    project = PROJECTS_DB[project_key]

    if not _is_user_authorized(member_id, role, project):
        logging.warning(
            f"Unauthorized attempt to edit project '{project_key}' settings "
            f"by user '{member_id}' with role '{role}'."
        )
        raise PermissionError("User not authorized to edit project settings.")

    _update_project_settings_in_db(project_key, project, updated_configuration)
    
    logging.info(
        f"Project '{project_key}' settings successfully updated by user '{member_id}'."
    )
    return "Project settings updated successfully."
