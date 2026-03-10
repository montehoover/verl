from typing import Dict, Any

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

def change_project_settings(owner_id: str, access_level: str, proj_id: str, modifications: Dict[str, Any]) -> str:
    """
    Allows authorized users to update project settings.

    Args:
        owner_id: The ID of the user attempting to update settings.
        access_level: The role of the user (e.g., 'admin', 'member').
        proj_id: The identifier of the project.
        modifications: A dictionary containing the new configuration settings.

    Returns:
        A success message if the settings are updated.

    Raises:
        ValueError: If the project ID is not found in PROJECTS_DB.
        PermissionError: If the user is not authorized to change settings.
    """
    if proj_id not in PROJECTS_DB:
        raise ValueError(f"Project with ID '{proj_id}' not found.")

    project = PROJECTS_DB[proj_id]

    # Authorization check: user must be the creator or have 'admin' access level
    is_creator = project.get("creator_id") == owner_id
    is_admin = access_level == "admin"

    if not (is_creator or is_admin):
        raise PermissionError(
            f"User '{owner_id}' with access level '{access_level}' is not authorized to change settings for project '{proj_id}'."
        )

    # Update the settings
    project["settings"].update(modifications)
    
    return f"Settings for project '{proj_id}' updated successfully."
