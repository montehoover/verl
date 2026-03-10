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


def _authorization_error(member_id: str, member_role: str, project: dict) -> str | None:
    """
    Pure function that determines if the member is authorized to modify a project's settings.

    Returns:
        None if authorized, otherwise a permission error message string.
    """
    members = project.get("members", [])
    if member_id not in members:
        return "Permission denied: user is not a member of the project."

    is_creator = member_id == project.get("creator_id")
    is_project_manager = member_role == "project_manager"

    if not (is_creator or is_project_manager):
        return "Permission denied: insufficient role to modify project settings."

    return None


def _merge_settings(current_settings: dict, fresh_settings: dict) -> dict:
    """
    Pure function that returns a new settings dict by shallow-merging fresh_settings
    over current_settings without mutating the inputs.
    """
    if not isinstance(current_settings, dict):
        current_settings = {}
    # Create a new dict to avoid mutating inputs
    merged = {**current_settings}
    merged.update(fresh_settings)
    return merged


def alter_project_preferences(member_id: str, member_role: str, proj_id: str, fresh_settings: dict) -> str:
    """
    Modify project settings if the user is authorized.

    Args:
        member_id: ID of the user attempting to modify the settings.
        member_role: Role of the user (e.g., 'project_manager', 'team_member', 'viewer').
        proj_id: ID of the project to update.
        fresh_settings: Dictionary of settings to apply.

    Returns:
        A success message string if the settings are updated.

    Raises:
        ValueError: If the project does not exist.
        TypeError: If fresh_settings is not a dict.
        PermissionError: If the user is not authorized to modify the settings.
    """
    if proj_id not in PROJECTS_DB:
        raise ValueError(f"Project '{proj_id}' not found.")

    if not isinstance(fresh_settings, dict):
        raise TypeError("fresh_settings must be a dict")

    project = PROJECTS_DB[proj_id]

    # Authorization using pure helper
    auth_err = _authorization_error(member_id, member_role, project)
    if auth_err is not None:
        raise PermissionError(auth_err)

    # Build updated settings using pure helper, then persist
    current_settings = project.get("settings", {})
    PROJECTS_DB[proj_id]["settings"] = _merge_settings(current_settings, fresh_settings)

    return f"Settings for project {proj_id} updated successfully."
