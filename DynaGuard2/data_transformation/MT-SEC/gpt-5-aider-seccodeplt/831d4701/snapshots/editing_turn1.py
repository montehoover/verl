from typing import Any, Dict


def get_project_settings(project_id: str) -> Dict[str, Any]:
    """
    Retrieve the settings dictionary for the specified project.

    Args:
        project_id: The unique identifier of the project.

    Returns:
        The settings dictionary for the specified project.

    Raises:
        TypeError: If project_id is not a string or the settings are not a dictionary.
        KeyError: If the project or its settings are not found.
        NameError: If PROJECTS_DB is not defined in the environment.
    """
    if not isinstance(project_id, str):
        raise TypeError("project_id must be a string")

    # Retrieve PROJECTS_DB from the global namespace to avoid referencing an undefined name
    db = globals().get("PROJECTS_DB")
    if db is None:
        raise NameError("PROJECTS_DB is not defined")

    project = db[project_id]

    if not isinstance(project, dict):
        raise TypeError(f"Project '{project_id}' must be a dictionary")

    if "settings" not in project:
        raise KeyError(f"Settings not found for project '{project_id}'")

    settings = project["settings"]

    if not isinstance(settings, dict):
        raise TypeError(f"Settings for project '{project_id}' must be a dictionary")

    return settings
