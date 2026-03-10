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


def change_project_config(uid: str, role: str, proj: str, settings_update: dict) -> str:
    """
    Update project settings if the user is authorized.

    Args:
        uid: The unique identifier of the user attempting the modification.
        role: The role of the user (e.g., 'project_manager', 'team_member', 'viewer').
        proj: The identifier of the project whose settings should be changed.
        settings_update: A dictionary of updates to apply to the project's settings.

    Returns:
        A success message upon successfully updating the project settings.

    Raises:
        PermissionError: If the user is not authorized to update the project settings.
        KeyError: If the specified project does not exist.
        TypeError: If settings_update is not a dict.
    """
    if proj not in PROJECTS_DB:
        raise KeyError(f"Project '{proj}' not found")

    if not isinstance(settings_update, dict):
        raise TypeError("settings_update must be a dict")

    project = PROJECTS_DB[proj]
    is_creator = uid == project.get("creator_id")
    is_member = uid in project.get("members", [])
    role_allowed = role in ("project_manager", "team_member")

    # Authorization:
    # - Creator can always update.
    # - Members with role 'project_manager' or 'team_member' can update.
    if not (is_creator or (is_member and role_allowed)):
        raise PermissionError("User is not authorized to modify project settings")

    # Apply updates to the project's settings
    project_settings = project.setdefault("settings", {})
    project_settings.update(settings_update)

    return f"Project settings for {proj} have been updated successfully"
