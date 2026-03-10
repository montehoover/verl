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


def update_project_configurations(uid: str, role: str, pid: str, settings: Dict[str, Any]) -> str:
    """
    Update the project's settings if the user is authorized.

    Authorization rules:
    - The project creator can always update.
    - Otherwise, the user must be a member of the project and have an allowed role ('admin' or 'owner').

    Args:
        uid: The unique identifier of the user.
        role: The user's role (e.g., 'admin', 'owner').
        pid: The unique identifier of the project.
        settings: A dictionary of settings to merge into the project's settings.

    Returns:
        A success message if the update is performed.

    Raises:
        TypeError: If argument types are invalid or project structure is malformed.
        KeyError: If the project is not found or settings missing in project.
        NameError: If PROJECTS_DB is not defined.
        PermissionError: If the user is not authorized to update the project settings.
    """
    if not isinstance(uid, str):
        raise TypeError("uid must be a string")
    if not isinstance(role, str):
        raise TypeError("role must be a string")
    if not isinstance(pid, str):
        raise TypeError("pid must be a string")
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dictionary")

    db = globals().get("PROJECTS_DB")
    if db is None:
        raise NameError("PROJECTS_DB is not defined")

    project = db[pid]

    if not isinstance(project, dict):
        raise TypeError(f"Project '{pid}' must be a dictionary")

    creator_id = project.get("creator_id")
    members = project.get("members", [])
    if not isinstance(members, (list, set, tuple)):
        raise TypeError(f"Project '{pid}' members must be a list, set, or tuple")

    # Authorization check
    allowed_roles = {"admin", "owner"}
    role_normalized = role.strip().lower()

    is_creator = uid == creator_id
    is_authorized_member = uid in members and role_normalized in allowed_roles

    if not (is_creator or is_authorized_member):
        raise PermissionError("Unauthorized: user is not allowed to update this project's settings")

    if "settings" not in project:
        raise KeyError(f"Settings not found for project '{pid}'")

    current_settings = project["settings"]
    if not isinstance(current_settings, dict):
        raise TypeError(f"Settings for project '{pid}' must be a dictionary")

    # Apply updates (shallow merge)
    current_settings.update(settings)

    return "Settings updated successfully"
