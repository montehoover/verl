from typing import Any, Dict


def get_project_settings(project_id: str) -> Dict[str, Any]:
    """
    Retrieve the current settings for a project.

    Args:
        project_id: The project ID.

    Returns:
        A dictionary of settings. If the project does not exist or has no settings,
        an empty dict is returned.

    Assumptions:
        - PROJECTS_DB is a dict available in the runtime environment.
        - PROJECTS_DB may map project_id -> {"settings": {...}} or directly to a settings dict.
        - If PROJECTS_DB is not present or not a dict, an empty dict is returned.
    """
    if not isinstance(project_id, str):
        raise TypeError("project_id must be a string")

    # Avoid referencing an undefined name; retrieve from globals safely.
    db = globals().get("PROJECTS_DB")
    if not isinstance(db, dict):
        return {}

    project_entry = db.get(project_id)
    if project_entry is None:
        return {}

    if isinstance(project_entry, dict):
        settings = project_entry.get("settings")
        if isinstance(settings, dict):
            return dict(settings)
        # Fallback: treat the project entry itself as the settings dict
        return dict(project_entry)

    return {}
