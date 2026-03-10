import logging

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

# Module-level logger for audit and debugging. Consumers can configure handlers/formatters.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def can_update_project_settings(owner_id: str, access_level: str, project: dict) -> bool:
    """
    Pure function to determine if a user can update project settings.

    Rules:
    - The project creator can always update settings.
    - A 'project_manager' who is also a member of the project can update settings.
    """
    role = (access_level or "").lower()
    is_creator = owner_id == project.get("creator_id")
    is_member = owner_id in project.get("members", [])
    return is_creator or (role == "project_manager" and is_member)


def merge_settings(current_settings: dict, modifications: dict) -> dict:
    """
    Pure function to return a new settings dict by merging modifications into current settings.
    Does not mutate the input dictionaries.
    """
    base = dict(current_settings or {})
    base.update(modifications or {})
    return base


def change_project_settings(owner_id: str, access_level: str, proj_id: str, modifications: dict) -> str:
    """
    Update project settings if the user has sufficient rights.

    Authorization rules:
    - The project creator can always update settings.
    - A user with access_level 'project_manager' who is a member of the project can update settings.

    Parameters:
        owner_id (str): The user attempting to update settings.
        access_level (str): The user's role (e.g., 'project_manager', 'team_member', 'viewer').
        proj_id (str): The identifier of the project.
        modifications (dict): New configuration to merge into the project's settings.

    Returns:
        str: Success message indicating the settings have been updated.

    Raises:
        KeyError: If the project does not exist.
        TypeError: If modifications is not a dict.
        PermissionError: If the user is not authorized to update settings.
    """
    logger.info(
        "settings_update_attempt owner_id=%s project_id=%s access_level=%s modifications=%r",
        owner_id, proj_id, access_level, modifications
    )

    project = PROJECTS_DB.get(proj_id)
    if project is None:
        logger.warning(
            "settings_update_failed reason=project_not_found owner_id=%s project_id=%s",
            owner_id, proj_id
        )
        raise KeyError(f"Project '{proj_id}' not found")

    if not isinstance(modifications, dict):
        logger.warning(
            "settings_update_failed reason=invalid_modifications_type owner_id=%s project_id=%s type=%s",
            owner_id, proj_id, type(modifications).__name__
        )
        raise TypeError("modifications must be a dict")

    if not can_update_project_settings(owner_id, access_level, project):
        logger.warning(
            "settings_update_failed reason=unauthorized owner_id=%s project_id=%s access_level=%s",
            owner_id, proj_id, access_level
        )
        raise PermissionError("Unauthorized access: insufficient rights to update project settings")

    # Merge modifications into existing settings without mutating inputs
    current = project.get("settings", {})
    new_settings = merge_settings(current, modifications)
    logger.debug(
        "settings_update_diff owner_id=%s project_id=%s before=%r after=%r",
        owner_id, proj_id, current, new_settings
    )

    project["settings"] = new_settings

    logger.info(
        "settings_update_success owner_id=%s project_id=%s applied_modifications=%r",
        owner_id, proj_id, modifications
    )
    return f"Settings for project {proj_id} have been updated."
