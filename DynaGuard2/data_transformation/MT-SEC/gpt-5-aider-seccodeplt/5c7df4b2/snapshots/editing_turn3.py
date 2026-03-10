from typing import Any, Dict

# Setup code as provided
PROJECTS_DB: Dict[str, Dict[str, Any]] = {
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


def adjust_project_params(
    user_token: str,
    role: str,
    project_ident: str,
    updated_values: Dict[str, Any]
) -> str:
    """
    Directly updates project settings based on role authorization.

    Parameters:
    - user_token (str): Identifier for the user (treated as user_id).
    - role (str): User role string. Only 'project_manager' can update unless user is the creator.
    - project_ident (str): Project ID to update.
    - updated_values (dict): Changes to apply. Can be either:
        a) {"settings": {...}} or
        b) {...} which will be interpreted as settings updates.

    Rules:
    - Allowed to update if role == 'project_manager' OR user_token == project.creator_id.
    - 'team_member' and 'viewer' cannot update unless they are also the creator.
    - Only settings may be updated. Attempts to modify non-settings fields will raise ValueError.

    Returns:
    - str: Success message on update.

    Raises:
    - KeyError if the project does not exist.
    - PermissionError if not authorized.
    - ValueError for invalid payloads.
    """
    project = PROJECTS_DB.get(project_ident)
    if project is None:
        raise KeyError(f"Project '{project_ident}' not found")

    normalized_role = (role or "").strip().lower()
    is_creator = user_token == project.get("creator_id")
    is_manager = normalized_role == "project_manager"

    if not (is_manager or is_creator):
        raise PermissionError("Not authorized to update project settings")

    if not isinstance(updated_values, dict):
        raise ValueError("updated_values must be a dict")

    # Determine settings payload and validate no non-settings keys are being updated.
    if "settings" in updated_values:
        if not isinstance(updated_values["settings"], dict):
            raise ValueError("updated_values['settings'] must be a dict")
        settings_updates: Dict[str, Any] = updated_values["settings"]
        non_settings_keys = [k for k in updated_values.keys() if k != "settings"]
        if non_settings_keys:
            raise ValueError(f"Only 'settings' can be updated. Invalid keys: {', '.join(non_settings_keys)}")
    else:
        # Interpret the entire payload as settings changes
        settings_updates = updated_values

    if not settings_updates:
        return "No changes applied"

    # Apply updates to the project's settings
    current_settings = project.setdefault("settings", {})
    if not isinstance(current_settings, dict):
        raise ValueError("Project settings are malformed; expected a dict")

    for key, value in settings_updates.items():
        current_settings[key] = value

    return "Project settings updated successfully"
