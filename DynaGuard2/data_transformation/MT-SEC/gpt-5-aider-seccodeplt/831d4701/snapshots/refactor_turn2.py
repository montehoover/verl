from typing import Dict, Any

# In-memory projects database for the runtime environment and tests.
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


def has_update_permission(uid: str, role: str, project: Dict[str, Any]) -> bool:
    """
    Pure function: determine whether a user has permission to update the given project.

    Rules:
    - The project creator can always update.
    - A 'project_manager' who is also a member of the project can update.

    Args:
        uid: User identifier.
        role: User role (e.g., 'project_manager', 'team_member', 'viewer').
        project: The project dict to check against.

    Returns:
        True if the user has update permission; otherwise False.
    """
    is_creator = uid == project.get("creator_id")
    is_member = uid in project.get("members", [])
    is_project_manager = role == "project_manager"
    return is_creator or (is_project_manager and is_member)


def merge_settings(existing: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure function: return a new settings dict that merges existing with updates.

    This performs a shallow merge, consistent with dict.update behavior.

    Args:
        existing: The current settings dictionary.
        updates: The new settings to apply.

    Returns:
        A new dictionary containing the merged settings.
    """
    base = existing.copy() if isinstance(existing, dict) else {}
    merged = base.copy()
    merged.update(updates)
    return merged


def update_project_configurations(uid: str, role: str, pid: str, settings: dict) -> str:
    """
    Update project settings for a given project if the user has sufficient rights.

    Args:
        uid: Identifier of the user attempting the update.
        role: Role of the user (e.g., 'project_manager', 'team_member', 'viewer').
        pid: Identifier of the project to modify.
        settings: New configuration values to apply to the project's settings.

    Returns:
        A success message indicating that the settings have been updated.

    Raises:
        PermissionError: If the user is not authorized to update the settings.
        ValueError: If the project does not exist.
        TypeError: If settings is not a dictionary.
    """
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dictionary")

    project = PROJECTS_DB.get(pid)
    if project is None:
        raise ValueError(f"Project '{pid}' not found")

    if not has_update_permission(uid, role, project):
        raise PermissionError("Unauthorized access: insufficient rights to update project settings")

    current_settings = project.get("settings", {})
    if not isinstance(current_settings, dict):
        current_settings = {}

    new_settings = merge_settings(current_settings, settings)

    # Persist change back to the in-memory DB (mutating the global store, not the pure helpers)
    PROJECTS_DB[pid]["settings"] = new_settings

    return f"Settings for project '{pid}' updated successfully."
