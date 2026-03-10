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
    Modify project settings if the user is authorized.

    Args:
        member_id: The ID of the user attempting the change.
        member_role: The user's role in the system ('project_manager', 'team_member', 'viewer').
        proj_id: The project ID to update.
        fresh_settings: A dictionary of settings to merge into the project's settings.

    Returns:
        A success message string when settings are updated.

    Raises:
        ValueError: If the project does not exist.
        PermissionError: If the user is not authorized to modify the project's settings.
    """
    # Validate project existence
    project = PROJECTS_DB.get(proj_id)
    if project is None:
        raise ValueError(f"Project {proj_id} not found")

    # Ensure the user is a member of the project
    if member_id not in project.get("members", []):
        raise PermissionError("User is not a member of this project")

    # Authorization rules:
    # - project_manager: can update settings
    # - team_member: can update only if they are also the project creator
    # - viewer or any other role: cannot update
    allowed = False
    if member_role == "project_manager":
        allowed = True
    elif member_role == "team_member" and member_id == project.get("creator_id"):
        allowed = True
    else:
        allowed = False

    if not allowed:
        raise PermissionError("You do not have permission to modify settings for this project")

    # Apply settings update
    if not isinstance(fresh_settings, dict):
        raise ValueError("fresh_settings must be a dictionary")

    project_settings = project.setdefault("settings", {})
    project_settings.update(fresh_settings)

    return f"Project settings for {proj_id} have been updated successfully"
