"""
Project settings management utilities with authorization and logging support.
"""

import logging


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


PROJECTS_DB = {
    "PROJ001": {
        "project_id": "PROJ001",
        "creator_id": "USER1",
        "members": ["USER1", "USER2", "USER3"],
        "settings": {"visibility": "private", "deadline": "2023-12-31"},
    },
    "PROJ002": {
        "project_id": "PROJ002",
        "creator_id": "USER4",
        "members": ["USER4", "USER5"],
        "settings": {"visibility": "public", "deadline": "2023-10-15"},
    },
}


def is_authorized_to_edit(member_id: str, role: str, project: dict) -> bool:
    """
    Determine if a user is authorized to edit project settings.

    Authorization policy:
    - Project creator can always update.
    - Project members with role 'project_manager' can update.
    - Other roles (e.g., 'team_member', 'viewer') cannot update.

    This function is pure and does not mutate the provided project.

    Args:
        member_id: The unique identifier of the user attempting the modification.
        role: The role of the user (e.g., 'project_manager', 'team_member',
            'viewer').
        project: The project dictionary from the PROJECTS_DB.

    Returns:
        True if authorized, False otherwise.
    """
    creator_id = project.get("creator_id")
    members = project.get("members", [])
    is_member = member_id in members or member_id == creator_id

    return (member_id == creator_id) or (is_member and role == "project_manager")


def apply_settings_update(
    current_settings: dict,
    updated_configuration: dict,
) -> dict:
    """
    Compute the updated settings without mutating inputs.

    Performs a shallow merge where keys in updated_configuration override those
    in current_settings. This function is pure and returns a new dictionary.

    Args:
        current_settings: Existing project settings.
        updated_configuration: New settings to apply.

    Returns:
        A new dict representing the merged settings.

    Raises:
        TypeError: If either argument is not a dict.
    """
    if not isinstance(current_settings, dict):
        raise TypeError("current_settings must be a dict")
    if not isinstance(updated_configuration, dict):
        raise TypeError("updated_configuration must be a dict")

    merged = current_settings.copy()
    merged.update(updated_configuration)
    return merged


def compute_settings_diff(old: dict, new: dict) -> dict:
    """
    Compute a diff between two settings dictionaries.

    The result is a dictionary with keys:
      - 'added': mapping of keys added in 'new' with their values.
      - 'removed': mapping of keys removed from 'old' with their old values.
      - 'updated': mapping of keys whose values changed with a dict containing
        'from' and 'to' values.

    This function is pure and does not mutate inputs.

    Args:
        old: The original settings.
        new: The updated settings.

    Returns:
        A dict describing added, removed, and updated keys.

    Raises:
        TypeError: If either argument is not a dict.
    """
    if not isinstance(old, dict):
        raise TypeError("old must be a dict")
    if not isinstance(new, dict):
        raise TypeError("new must be a dict")

    old_keys = set(old.keys())
    new_keys = set(new.keys())

    added_keys = new_keys - old_keys
    removed_keys = old_keys - new_keys
    common_keys = old_keys & new_keys

    updated_keys = {k for k in common_keys if old[k] != new[k]}

    added = {k: new[k] for k in added_keys}
    removed = {k: old[k] for k in removed_keys}
    updated = {k: {"from": old[k], "to": new[k]} for k in updated_keys}

    return {"added": added, "removed": removed, "updated": updated}


def edit_project_settings(
    member_id: str,
    role: str,
    project_key: str,
    updated_configuration: dict,
) -> str:
    """
    Update project settings if the user is authorized.

    This function validates inputs, checks authorization, applies the settings
    update, persists it to the in-memory database, and emits log entries
    that record the attempted and successful updates.

    Args:
        member_id: The unique identifier of the user attempting the
            modification.
        role: The role of the user (e.g., 'project_manager', 'team_member',
            'viewer').
        project_key: The identifier of the project to update.
        updated_configuration: The updated settings for the project.

    Returns:
        A success message upon successfully updating the project settings.

    Raises:
        PermissionError: If the user is not authorized to update the settings.
        KeyError: If the project does not exist.
        TypeError: If updated_configuration is not a dict.
    """
    if not isinstance(updated_configuration, dict):
        raise TypeError("updated_configuration must be a dict")

    project = PROJECTS_DB.get(project_key)
    if project is None:
        logger.warning(
            "Project not found for update attempt: project_key=%s, "
            "member_id=%s, role=%s",
            project_key,
            member_id,
            role,
        )
        raise KeyError(f"Project '{project_key}' does not exist")

    if not is_authorized_to_edit(member_id, role, project):
        logger.warning(
            "Unauthorized settings update attempt: project_key=%s, "
            "member_id=%s, role=%s",
            project_key,
            member_id,
            role,
        )
        raise PermissionError(
            "You do not have permission to update project settings"
        )

    current_settings = project.get("settings", {})
    new_settings = apply_settings_update(current_settings, updated_configuration)
    diff = compute_settings_diff(current_settings, new_settings)

    # Persist changes
    PROJECTS_DB[project_key]["settings"] = new_settings

    # Log the changes: only log keys and values, avoiding sensitive payloads.
    if not (diff["added"] or diff["removed"] or diff["updated"]):
        logger.info(
            "No settings changes detected for project_key=%s by member_id=%s "
            "(role=%s)",
            project_key,
            member_id,
            role,
        )
    else:
        logger.info(
            "Project settings updated: project_key=%s, by member_id=%s "
            "(role=%s), changes=%s",
            project_key,
            member_id,
            role,
            diff,
        )

    return f"Project {project_key} settings updated successfully"
