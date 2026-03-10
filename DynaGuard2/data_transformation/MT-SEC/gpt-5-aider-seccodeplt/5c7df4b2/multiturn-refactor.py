import logging
import json

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


def _get_audit_logger() -> logging.Logger:
    """
    Configure and return a module-level audit logger that writes to a file
    in the current directory. Ensures handlers are added only once.
    """
    logger = logging.getLogger("project_audit")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler("project_audit.log", encoding="utf-8")
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def _to_json_safe(obj) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        # Fall back to repr if the object is not JSON-serializable
        return repr(obj)


def check_permissions(user_token: str, role: str, project: dict) -> bool:
    """
    Pure function: Determine whether a user is authorized to update project settings.

    Authorization policy:
    - The project creator can always update settings.
    - A user with role 'project_manager' who is a member of the project can update settings.
    - Other roles (e.g., 'team_member', 'viewer') are not permitted to update project settings.
    """
    if not isinstance(project, dict):
        return False

    creator_id = project.get("creator_id")
    members = set(project.get("members", []))
    normalized_role = role.lower().strip() if isinstance(role, str) else ""

    if user_token == creator_id:
        return True

    if normalized_role == "project_manager" and user_token in members:
        return True

    return False


def build_updated_settings(existing_settings: dict, updated_values: dict) -> dict:
    """
    Pure function: Return a new settings dict by merging updated_values into existing_settings.
    Does not mutate the inputs.
    """
    base = dict(existing_settings) if isinstance(existing_settings, dict) else {}
    if not isinstance(updated_values, dict):
        raise TypeError("updated_values must be a dict.")
    base.update(updated_values)
    return base


def adjust_project_params(user_token: str, role: str, project_ident: str, updated_values: dict) -> str:
    """
    Update project settings if the user has sufficient rights.

    Args:
        user_token: The identifier representing the user attempting to update settings.
        role: The user's role (e.g., 'project_manager', 'team_member', 'viewer').
        project_ident: The identifier of the project being modified.
        updated_values: A dict containing the new configuration for the project.

    Returns:
        A success message indicating that the settings have been updated.

    Raises:
        PermissionError: If the user is not authorized to update project settings.
        KeyError: If the project_ident does not correspond to a known project.
        TypeError: If provided arguments are of incorrect types.
    """
    logger = _get_audit_logger()
    logger.info(
        "ATTEMPT action=update_settings user=%r role=%r project=%r updates=%s",
        user_token,
        role,
        project_ident,
        _to_json_safe(updated_values),
    )

    # Type validation with audit logs for failures
    if not isinstance(user_token, str):
        logger.error(
            "FAILED action=update_settings user=%r role=%r project=%r reason=type_error field=user_token",
            user_token, role, project_ident
        )
        raise TypeError("user_token must be a string.")
    if not isinstance(role, str):
        logger.error(
            "FAILED action=update_settings user=%r role=%r project=%r reason=type_error field=role",
            user_token, role, project_ident
        )
        raise TypeError("role must be a string.")
    if not isinstance(project_ident, str):
        logger.error(
            "FAILED action=update_settings user=%r role=%r project=%r reason=type_error field=project_ident",
            user_token, role, project_ident
        )
        raise TypeError("project_ident must be a string.")
    if not isinstance(updated_values, dict):
        logger.error(
            "FAILED action=update_settings user=%r role=%r project=%r reason=type_error field=updated_values",
            user_token, role, project_ident
        )
        raise TypeError("updated_values must be a dict.")

    if project_ident not in PROJECTS_DB:
        logger.error(
            "FAILED action=update_settings user=%r role=%r project=%r reason=project_not_found",
            user_token, role, project_ident
        )
        raise KeyError(f"Project '{project_ident}' not found.")

    project = PROJECTS_DB[project_ident]
    before_settings = project.get("settings", {}) or {}

    if not check_permissions(user_token, role, project):
        logger.warning(
            "DENIED action=update_settings user=%r role=%r project=%r reason=insufficient_permissions",
            user_token, role, project_ident
        )
        raise PermissionError("Unauthorized access: insufficient permissions to update project settings.")

    new_settings = build_updated_settings(before_settings, updated_values)
    PROJECTS_DB[project_ident]["settings"] = new_settings

    logger.info(
        "SUCCESS action=update_settings user=%r role=%r project=%r changed_keys=%s before=%s after=%s",
        user_token,
        role,
        project_ident,
        _to_json_safe(list(updated_values.keys())),
        _to_json_safe(before_settings),
        _to_json_safe(new_settings),
    )

    return f"Settings for project {project_ident} have been updated successfully."
