from typing import Any, Dict
from collections.abc import Mapping


def get_project_settings(project_key: str) -> Dict[str, Any]:
    """
    Retrieve the settings dictionary for the specified project.

    Assumes a global PROJECTS_DB is available in module globals().

    Behavior:
    - If PROJECTS_DB[project_key] is a dict and contains a 'settings' dict, return that.
    - Else if PROJECTS_DB[project_key] is a dict, return it (assumed to be the settings).
    - Else if the project object has a 'settings' attribute that is a dict, return it.
    - Raises KeyError if the project_key is not found.
    - Raises TypeError for invalid input or if settings are not in a dictionary form.

    :param project_key: The unique key (string) identifying the project.
    :return: The settings dictionary for the specified project.
    """
    if not isinstance(project_key, str):
        raise TypeError("project_key must be a string")

    db = globals().get("PROJECTS_DB")
    if db is None:
        raise NameError("PROJECTS_DB is not defined")
    if not isinstance(db, Mapping):
        raise TypeError("PROJECTS_DB must be a mapping/dictionary")

    try:
        project = db[project_key]
    except KeyError as exc:
        raise KeyError(f"Project with key '{project_key}' not found") from exc

    # Case 1: Conventional structure: {"settings": {...}}
    if isinstance(project, dict) and "settings" in project:
        settings = project["settings"]
        if isinstance(settings, dict):
            return settings
        raise TypeError("Project 'settings' must be a dictionary")

    # Case 2: The project entry itself is the settings dict
    if isinstance(project, dict):
        return project

    # Case 3: Object with a 'settings' attribute
    if hasattr(project, "settings"):
        settings = getattr(project, "settings")
        if isinstance(settings, dict):
            return settings
        raise TypeError("Project 'settings' attribute must be a dictionary")

    raise TypeError("Project settings not found or not in a dictionary form")
