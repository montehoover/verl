from typing import Dict, Any

# In-memory projects database for the runtime environment and tests.
PROJECTS_DB: Dict[str, Dict[str, Any]] = {
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


def update_project_configurations(uid: str, role: str, pid: str, settings: dict) -> str:
    """
    Update project settings for a given project if the user has sufficient rights.

    Args:
        uid: Identifier of the user attempting the update.
        role: Role of the user (e.g., 'project_manager', 'team_member', 'viewer').
        pid: Identifier of the project to modify.
        settings: New configuration values to apply to the project's settings.

    Returns:
        A success message indicating that the settings have been updated.

    Raises:
        PermissionError: If the user is not authorized to update the settings.
        ValueError: If the project does not exist.
        TypeError: If settings is not a dictionary.
    """
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dictionary")

    project = PROJECTS_DB.get(pid)
    if project is None:
        raise ValueError(f"Project '{pid}' not found")

    # Authorization rules:
    # - The project creator can always update.
    # - A project_manager who is a member of the project can update.
    is_creator = uid == project.get("creator_id")
    is_member = uid in project.get("members", [])
    is_project_manager = role == "project_manager"

    if not (is_creator or (is_project_manager and is_member)):
        raise PermissionError("Unauthorized access: insufficient rights to update project settings")

    if "settings" not in project or not isinstance(project["settings"], dict):
        project["settings"] = {}

    project["settings"].update(settings)

    # Persist change back to the in-memory DB
    PROJECTS_DB[pid] = project

    return f"Settings for project '{pid}' updated successfully."
