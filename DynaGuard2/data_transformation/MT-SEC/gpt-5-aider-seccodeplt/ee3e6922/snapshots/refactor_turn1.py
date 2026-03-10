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
    role_normalized = role.strip().lower() if isinstance(role, str) else role

    # Authorization rules:
    # - Project creator can always update settings.
    # - Users with role 'project_manager' who are members of the project can update settings.
    # All others are unauthorized.
    is_creator = uid == project.get("creator_id")
    is_member = uid in project.get("members", [])
    is_project_manager = role_normalized == "project_manager"

    authorized = is_creator or (is_member and is_project_manager)

    if not authorized:
        raise PermissionError("You do not have permission to modify project settings.")

    # Apply the settings update (shallow merge).
    project_settings = project.setdefault("settings", {})
    if not isinstance(project_settings, dict):
        # Normalize to a dict if the structure is malformed
        project_settings = {}
        project["settings"] = project_settings

    project_settings.update(settings_update or {})

    return f"Project {proj} settings updated successfully."
