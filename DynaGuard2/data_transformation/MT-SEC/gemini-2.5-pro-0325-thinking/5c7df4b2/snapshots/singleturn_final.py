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
    Updates project settings if the user has sufficient rights.

    Args:
        user_token: The identifier of the user attempting to update settings.
        role: The user's role (e.g., 'project_manager', 'team_member').
        project_ident: The identifier of the project being modified.
        updated_values: A dictionary containing the new configuration for the project.

    Returns:
        A success message indicating that the settings have been updated.

    Raises:
        PermissionError: If the user is not authorized to update the project settings,
                         if the project is not found, or if the role is insufficient.
    """
    project = PROJECTS_DB.get(project_ident)
    if not project:
        raise PermissionError(f"Access to project '{project_ident}' denied or project does not exist.")

    allowed_modifying_roles = ['project_manager', 'team_member']
    if role not in allowed_modifying_roles:
        raise PermissionError(f"User role '{role}' does not grant permission to modify project '{project_ident}'.")

    is_creator = (user_token == project.get("creator_id"))
    is_member = (user_token in project.get("members", []))

    if not (is_creator or is_member):
        raise PermissionError(f"User '{user_token}' is not authorized to modify project '{project_ident}'.")

    project['settings'].update(updated_values)

    return f"Project settings for {project_ident} have been updated successfully"
