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

    # Authorization check:
    # User must be a 'project_manager' OR the creator of the project.
    is_project_manager = (member_role == 'project_manager')
    is_creator = (member_id == project_data['creator_id'])

    if not (is_project_manager or is_creator):
        raise PermissionError(
            f"User '{member_id}' with role '{member_role}' does not have permission to alter settings for project '{proj_id}'."
        )

    # If authorized, update the settings.
    # The problem implies 'settings' key always exists in a valid project entry.
    project_data['settings'].update(fresh_settings)

    return f"Settings for project '{proj_id}' updated successfully."
