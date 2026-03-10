from typing import Any, Dict


def get_project_settings(project_id: str, user_id: str) -> Dict[str, Any]:
    """
    Retrieve the current settings for a project if the user has access.

    Args:
        project_id: The project ID.
        user_id: The requesting user's ID.

    Returns:
        A dictionary of settings if the user is a member of the project.
        If the project does not exist or has no settings, an empty dict is returned.
        If the user lacks permissions, a dict with an "error" key is returned.

    Assumptions:
        - PROJECTS_DB is a dict available in the runtime environment.
        - PROJECTS_DB may map:
            project_id -> {
                "settings": {...},
                "members": [<user_id>, ...]  # list/tuple/set of user IDs
            }
          or directly to a settings dict (in which case access is denied unless
          membership info is available).
        - If PROJECTS_DB is not present or not a dict, an empty dict is returned.
    """
    if not isinstance(project_id, str):
        raise TypeError("project_id must be a string")
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a string")

    # Avoid referencing an undefined name; retrieve from globals safely.
    db = globals().get("PROJECTS_DB")
    if not isinstance(db, dict):
        return {}

    project_entry = db.get(project_id)
    if project_entry is None:
        return {}

    # Require membership info for authorization.
    if isinstance(project_entry, dict):
        members = project_entry.get("members")
        is_member = isinstance(members, (list, set, tuple)) and user_id in members
        if not is_member:
            return {"error": "Insufficient permissions"}

        settings = project_entry.get("settings")
        if isinstance(settings, dict):
            return dict(settings)

        # Fallback: treat the project entry itself as the settings dict,
        # but do not leak membership lists or non-setting metadata.
        result = dict(project_entry)
        result.pop("members", None)
        return result

    # Unknown project schema; deny access without clear membership data.
    return {"error": "Insufficient permissions"}


def change_project_settings(
    owner_id: str,
    access_level: str,
    proj_id: str,
    modifications: Dict[str, Any],
) -> str:
    """
    Update the settings of a project if the user is authorized.

    Args:
        owner_id: The ID of the user attempting the update.
        access_level: The user's role or access level (e.g., 'admin', 'owner').
        proj_id: The project identifier.
        modifications: The new configuration to apply to the project's settings.

    Returns:
        A success message if the settings are updated.

    Raises:
        PermissionError: If the user is not authorized to update the project.
        KeyError: If the project does not exist.
        TypeError: If argument types are invalid.
    """
    if not isinstance(owner_id, str):
        raise TypeError("owner_id must be a string")
    if not isinstance(access_level, str):
        raise TypeError("access_level must be a string")
    if not isinstance(proj_id, str):
        raise TypeError("proj_id must be a string")
    if not isinstance(modifications, dict):
        raise TypeError("modifications must be a dict")

    db = globals().get("PROJECTS_DB")
    if not isinstance(db, dict):
        raise RuntimeError("PROJECTS_DB is not available")

    project = db.get(proj_id)
    if project is None:
        raise KeyError(f"Project '{proj_id}' not found")

    if not isinstance(project, dict):
        raise KeyError(f"Project '{proj_id}' is in an invalid format")

    members = project.get("members")
    creator_id = project.get("creator_id")

    is_member = isinstance(members, (list, set, tuple)) and owner_id in members
    # Define roles that are allowed to modify settings (in addition to the creator).
    allowed_roles = {"admin", "owner"}
    role_allows = access_level.lower() in allowed_roles

    authorized = (owner_id == creator_id) or (is_member and role_allows)
    if not authorized:
        raise PermissionError("Insufficient permissions")

    settings = project.get("settings")
    if not isinstance(settings, dict):
        settings = {}
        project["settings"] = settings

    # Apply shallow update of settings.
    settings.update(modifications)

    return "Settings updated successfully"
