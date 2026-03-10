from typing import Dict, Any


def get_project_settings(project_id: str) -> Dict[str, Any]:
    """
    Retrieve the settings dictionary for the given project ID from PROJECTS_DB.

    Assumptions:
    - PROJECTS_DB is a global dict-like object available in the runtime environment.
    - Each project entry in PROJECTS_DB is either:
        a) the settings dictionary itself, or
        b) a dictionary containing a 'settings' key whose value is the settings dictionary.

    Raises:
        TypeError: If project_id is not a string.
        KeyError: If the project_id does not exist in PROJECTS_DB.
        ValueError: If the project entry exists but does not contain a settings dictionary.
        NameError: If PROJECTS_DB is not defined.

    Returns:
        Dict[str, Any]: The settings dictionary for the given project.
    """
    if not isinstance(project_id, str):
        raise TypeError("project_id must be a string")

    try:
        db = PROJECTS_DB  # type: ignore[name-defined]
    except NameError as e:
        raise NameError("PROJECTS_DB is not defined in the current environment") from e

    project_entry = db[project_id]

    if isinstance(project_entry, dict):
        # Case where the project entry contains a nested 'settings' dict
        if "settings" in project_entry and isinstance(project_entry["settings"], dict):
            return project_entry["settings"]
        # Case where the project entry itself is the settings dict
        return project_entry

    raise ValueError(f"Project entry for '{project_id}' does not contain a settings dictionary")


def is_member_of_project(user_id: str, project_id: str) -> bool:
    """
    Check whether the given user is a member of the specified project.

    Assumptions:
    - PROJECTS_DB is a global dict-like object available in the runtime environment.
    - Each project entry in PROJECTS_DB is a dictionary that may contain:
        - a 'members' key whose value is an iterable (list, set, tuple) of user IDs, or
        - a 'members' key whose value is a dict keyed by user IDs.

    Returns:
        True if the user is a member of the project, otherwise False.

    Notes:
        - If PROJECTS_DB is not defined, the project does not exist, or the project entry
          does not contain recognizable membership data, this function returns False.
    """
    if not isinstance(user_id, str):
        raise TypeError("user_id must be a string")
    if not isinstance(project_id, str):
        raise TypeError("project_id must be a string")

    try:
        db = PROJECTS_DB  # type: ignore[name-defined]
    except NameError:
        return False

    project_entry = db.get(project_id)
    if not isinstance(project_entry, dict):
        return False

    members = project_entry.get("members")

    # If members is a dict, use keys as user IDs.
    if isinstance(members, dict):
        return user_id in members

    # If members is a common iterable of user IDs.
    if isinstance(members, (list, set, tuple)):
        return user_id in members

    return False
