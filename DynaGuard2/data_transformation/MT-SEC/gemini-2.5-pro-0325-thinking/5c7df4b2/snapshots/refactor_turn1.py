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
    project = PROJECTS_DB.get(project_ident)

    if not project:
        raise ValueError(f"Project with ID '{project_ident}' not found.")

    is_creator = project["creator_id"] == user_token
    is_project_manager = role == 'project_manager'

    if not (is_creator or is_project_manager):
        raise PermissionError(f"User '{user_token}' with role '{role}' is not authorized to update project '{project_ident}'.")

    project["settings"].update(updated_values)
    return f"Settings for project '{project_ident}' updated successfully."
