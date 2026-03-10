from typing import Dict, Any
import logging

# Logger setup for this module. By default, we do not configure handlers here,
# allowing the host application to configure logging as desired.
logger = logging.getLogger(__name__)

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


def compute_settings_diff(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure function: compute a shallow diff between two settings dictionaries.

    Returns a dictionary with three keys:
      - 'added': mapping of keys added in 'new' with their values
      - 'removed': mapping of keys removed in 'new' with their old values
      - 'changed': mapping of keys that exist in both but changed, as {'from': old, 'to': new}
    """
    old_keys = set(old.keys()) if isinstance(old, dict) else set()
    new_keys = set(new.keys()) if isinstance(new, dict) else set()

    added = {k: new[k] for k in (new_keys - old_keys)}
    removed = {k: old[k] for k in (old_keys - new_keys)}
    changed = {k: {"from": old[k], "to": new[k]}
               for k in (old_keys & new_keys)
               if old.get(k) != new.get(k)}

    return {"added": added, "removed": removed, "changed": changed}


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

    logger.debug("Received settings update request: pid=%s, uid=%s, role=%s, keys=%s",
                 pid, uid, role, list(settings.keys()))

    project = PROJECTS_DB.get(pid)
    if project is None:
        logger.warning("Settings update failed: project not found. pid=%s, uid=%s, role=%s",
                       pid, uid, role)
        raise ValueError(f"Project '{pid}' not found")

    if not has_update_permission(uid, role, project):
        logger.warning("Unauthorized settings update attempt: pid=%s, uid=%s, role=%s, attempted_keys=%s",
                       pid, uid, role, list(settings.keys()))
        raise PermissionError("Unauthorized access: insufficient rights to update project settings")

    current_settings = project.get("settings", {})
    if not isinstance(current_settings, dict):
        current_settings = {}

    new_settings = merge_settings(current_settings, settings)
    diff = compute_settings_diff(current_settings, new_settings)

    # Persist change back to the in-memory DB (mutating the global store, not the pure helpers)
    PROJECTS_DB[pid]["settings"] = new_settings

    logger.info("Settings update successful: pid=%s, uid=%s, role=%s, added_keys=%s, changed_keys=%s, removed_keys=%s",
                pid, uid, role,
                list(diff["added"].keys()),
                list(diff["changed"].keys()),
                list(diff["removed"].keys()))
    logger.debug("Settings diff details for pid=%s: %s", pid, diff)

    return f"Settings for project '{pid}' updated successfully."
