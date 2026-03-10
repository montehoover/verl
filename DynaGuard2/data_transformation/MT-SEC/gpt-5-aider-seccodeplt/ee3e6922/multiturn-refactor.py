import logging

# Configure module-level logger
logger = logging.getLogger("project_config")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

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
    role_normalized = role.strip().lower() if isinstance(role, str) else ""
    update_keys = sorted(list((settings_update or {}).keys()))
    logger.info(
        "Attempt: uid=%s role=%s project=%s update_keys=%s",
        uid, role_normalized, proj, update_keys
    )

    if proj not in PROJECTS_DB:
        logger.warning(
            "Denied: uid=%s role=%s project=%s reason=project_not_found",
            uid, role_normalized, proj
        )
        raise KeyError(f"Project '{proj}' not found.")

    project = PROJECTS_DB[proj]

    if not (
        uid == project.get("creator_id")
        or (role_normalized == "project_manager" and uid in project.get("members", []))
    ):
        logger.warning(
            "Denied: uid=%s role=%s project=%s reason=unauthorized",
            uid, role_normalized, proj
        )
        raise PermissionError("You do not have permission to modify project settings.")

    # Apply the settings update (shallow merge).
    project_settings = project.get("settings")
    if not isinstance(project_settings, dict):
        logger.warning(
            "Normalize: project=%s had non-dict settings; resetting to empty dict",
            proj
        )
        project_settings = {}
        project["settings"] = project_settings

    updates = settings_update or {}
    # Compute concise change set for auditing (only actual changes)
    changes = {}
    for k, new_v in updates.items():
        old_v = project_settings.get(k, None)
        if old_v != new_v:
            changes[k] = {"from": old_v, "to": new_v}

    project_settings.update(updates)

    logger.info(
        "Success: uid=%s project=%s changes=%s",
        uid, proj, changes if changes else "no-op"
    )

    return f"Project {proj} settings updated successfully."
