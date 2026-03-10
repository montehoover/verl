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

def edit_project_settings(member_id: str, role: str, project_key: str, updated_configuration: dict) -> str:
    """
    Allows users to update project settings.

    Args:
        member_id: The unique identifier of the user attempting the modification.
        role: The user's role.
        project_key: The identifier of the project.
        updated_configuration: The new settings for the project.

    Returns:
        A success message if the update is successful.

    Raises:
        ValueError: If the project_key is not found in PROJECTS_DB.
        PermissionError: If the user is not authorized to edit the project settings.
    """
    if project_key not in PROJECTS_DB:
        raise ValueError(f"Project with key '{project_key}' not found.")

    project = PROJECTS_DB[project_key]

    # Check for permission: user must be the creator or have an 'admin' role.
    # Assumes 'admin' role grants universal edit rights for simplicity.
    # More complex role/permission logic could be implemented here.
    if member_id == project["creator_id"] or role == "admin":
        project["settings"].update(updated_configuration)
        return f"Project '{project_key}' settings updated successfully."
    else:
        raise PermissionError(f"User '{member_id}' with role '{role}' is not authorized to edit settings for project '{project_key}'.")
