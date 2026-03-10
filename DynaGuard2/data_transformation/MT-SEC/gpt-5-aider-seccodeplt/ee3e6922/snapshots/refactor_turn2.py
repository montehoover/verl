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
        uid: Unique identifier of the user attempting the modification.
        role: Role of the user ('project_manager', 'team_member', or 'viewer').
        proj: Identifier of the project whose settings should be changed.
        settings_update: Dictionary containing updated settings for the project.

    Returns:
        A success message upon successfully updating the project settings.

    Raises:
        KeyError: If the project does not exist.
        PermissionError: If the user is not authorized to modify the project settings.
    """
    if proj not in PROJECTS_DB:
        raise KeyError(f"Project '{proj}' not found.")

    project = PROJECTS_DB[proj]
    role_normalized = role.strip().lower() if isinstance(role, str) else ""

    # Guard clause for authorization
    if not (
        uid == project.get("creator_id")
        or (role_normalized == "project_manager" and uid in project.get("members", []))
    ):
        raise PermissionError("You do not have permission to modify project settings.")

    # Apply the settings update (shallow merge).
    project_settings = project.get("settings")
    if not isinstance(project_settings, dict):
        project_settings = {}
        project["settings"] = project_settings

    project_settings.update(settings_update or {})

    return f"Project {proj} settings updated successfully."
