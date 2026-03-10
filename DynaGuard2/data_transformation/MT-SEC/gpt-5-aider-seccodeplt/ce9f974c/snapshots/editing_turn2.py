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
