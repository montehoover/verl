from typing import Any, Dict, Union


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


def get_authorized_project_settings(user_id: str, project_id: str) -> Union[Dict[str, Any], str]:
    """
    Return project settings if the user is a member of the project; otherwise return an unauthorized message.

    Args:
        user_id: The unique identifier of the user.
        project_id: The unique identifier of the project.

    Returns:
        The settings dictionary for the specified project if the user is a member,
        otherwise a string with an unauthorized access message.

    Raises:
        TypeError: If user_id or project_id is not a string, or invalid project structure.
        KeyError: If the project is not found.
        NameError: If PROJECTS_DB is not defined in the environment.
    """
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a string")
    if not isinstance(project_id, str):
        raise TypeError("project_id must be a string")

    db = globals().get("PROJECTS_DB")
    if db is None:
        raise NameError("PROJECTS_DB is not defined")

    project = db[project_id]

    if not isinstance(project, dict):
        raise TypeError(f"Project '{project_id}' must be a dictionary")

    members = project.get("members", [])
    if not isinstance(members, (list, set, tuple)):
        raise TypeError(f"Project '{project_id}' members must be a list, set, or tuple")

    if user_id not in members:
        return "Unauthorized: user is not a member of this project"

    return get_project_settings(project_id)
