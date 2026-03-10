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


def edit_project_settings(member_id: str, role: str, project_key: str, updated_configuration: dict) -> str:
    """
    Update the settings for a project if the member is authorized.

    Parameters:
        member_id (str): the unique identifier of the user attempting the modification
        role (str): 'project_manager', 'team_member', or 'viewer'
        project_key (str): the identifier of the project whose settings should be changed
        updated_configuration (dict): the updated settings for the project

    Returns:
        str: success message upon successfully updating the project settings

    Raises:
        PermissionError: for an unauthorized access
        KeyError: if the project_key does not exist
        TypeError: if updated_configuration or project settings are not dictionaries
    """
    if not isinstance(updated_configuration, dict):
        raise TypeError("updated_configuration must be a dict")

    project = PROJECTS_DB.get(project_key)
    if project is None:
        raise KeyError(f"Project '{project_key}' not found")

    normalized_role = (role or "").strip().lower()
    is_creator = member_id == project.get("creator_id")
    is_member = member_id in project.get("members", [])

    # Authorization rules:
    # - Creator can always update.
    # - Project members with role 'project_manager' or 'team_member' can update.
    # - 'viewer' cannot update even if a member.
    authorized_roles = {"project_manager", "team_member"}

    if not (is_creator or (is_member and normalized_role in authorized_roles)):
        raise PermissionError(
            f"User '{member_id}' is not authorized to update settings for project '{project_key}'"
        )

    settings = project.setdefault("settings", {})
    if not isinstance(settings, dict):
        raise TypeError("Project settings must be a dict")

    settings.update(updated_configuration)

    return f"Project settings for {project_key} have been updated successfully"
