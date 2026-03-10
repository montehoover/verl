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

def update_project_configurations(uid: str, role: str, pid: str, settings: dict) -> str:
    """
    Allows authorized users to update project settings.

    Args:
        uid: The user ID of the person attempting the update.
        role: The role of the user (e.g., 'admin', 'member').
        pid: The project ID of the project to update.
        settings: A dictionary containing the new settings to apply.

    Returns:
        A success message if the settings are updated.

    Raises:
        PermissionError: If the user is not authorized to update the settings
                         or if the project is not found.
        ValueError: If the project ID is not found in PROJECTS_DB.
    """
    if pid not in PROJECTS_DB:
        raise ValueError(f"Project '{pid}' not found.")

    project = PROJECTS_DB[pid]
    
    is_creator = (project["creator_id"] == uid)
    # Assuming 'admin' role is required for non-creator members to update settings.
    # This can be adjusted based on specific role management rules.
    is_authorized_member = (uid in project["members"] and role == "admin")

    if not (is_creator or is_authorized_member):
        raise PermissionError(
            f"User '{uid}' with role '{role}' is not authorized to update settings for project '{pid}'."
        )

    # Update existing settings with new values; new keys will be added.
    project["settings"].update(settings)
    
    return f"Settings for project '{pid}' updated successfully by user '{uid}'."
