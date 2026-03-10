from datetime import datetime

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

ACCESS_LOG = []

ALLOWED_ROLES = {"project_manager", "team_member"}


def log_project_access(uid: str, role: str, proj: str):
    """
    Check if a user can view a project's settings based on their role and log the access attempt.

    Args:
        uid: User's unique identifier.
        role: User's role (e.g., 'project_manager', 'team_member', 'viewer').
        proj: Project identifier (e.g., 'PROJ001').

    Returns:
        The project's settings dict if access is allowed.

    Raises:
        KeyError: If the project does not exist.
        PermissionError: If the user's role is not permitted to view settings.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    log_entry = {
        "timestamp": timestamp,
        "uid": uid,
        "role": role,
        "project_id": proj,
    }

    if proj not in PROJECTS_DB:
        log_entry["allowed"] = False
        log_entry["reason"] = "project_not_found"
        ACCESS_LOG.append(log_entry)
        raise KeyError(f"Project '{proj}' not found")

    allowed = role in ALLOWED_ROLES
    log_entry["allowed"] = allowed

    if allowed:
        log_entry["reason"] = "ok"
        ACCESS_LOG.append(log_entry)
        return PROJECTS_DB[proj]["settings"]

    log_entry["reason"] = "forbidden_role"
    ACCESS_LOG.append(log_entry)
    raise PermissionError(
        f"User '{uid}' with role '{role}' cannot view settings for project '{proj}'"
    )
