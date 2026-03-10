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


def authenticate_user(member_id: str, password: str) -> bool:
    """
    Authenticates a user based on member_id and password.

    Args:
        member_id: The user's member ID.
        password: The user's password.

    Returns:
        True if the credentials are correct, False otherwise.
    """
    # In a real application, this would involve checking credentials
    # against a database or other secure storage.
    # For this example, we'll use a hardcoded credential.
    # IMPORTANT: Do NOT use hardcoded credentials in production code.
    if member_id == "user123" and password == "securepassword123":
        return True
    return False


def get_project_settings(project_key: str, member_id: str) -> dict | None:
    """
    Retrieves project settings if the member is part of the project.

    Args:
        project_key: The unique key for the project.
        member_id: The ID of the member requesting the settings.

    Returns:
        A dictionary containing project settings if the member is authorized,
        otherwise None.
    """
    # Placeholder for project data and memberships
    # In a real application, this data would come from a database.
    projects_data = {
        "projectA": {
            "members": ["user123", "user456"],
            "settings": {"name": "Project Alpha", "status": "active", "budget": 10000}
        },
        "projectB": {
            "members": ["user789"],
            "settings": {"name": "Project Beta", "status": "pending", "budget": 5000}
        }
    }

    project = projects_data.get(project_key)
    if project and member_id in project["members"]:
        return project["settings"]
    return None


def edit_project_settings(member_id: str, role: str, project_key: str, updated_configuration: dict) -> str:
    """
    Allows authorized users to update project settings.

    Args:
        member_id: The ID of the member attempting the edit.
        role: The role of the member (e.g., 'admin', 'member').
        project_key: The unique key for the project.
        updated_configuration: A dictionary with settings to update.

    Returns:
        A success message if the operation is successful.

    Raises:
        PermissionError: If the user is not authorized to edit settings.
        ValueError: If the project_key does not exist.
    """
    project = PROJECTS_DB.get(project_key)

    if not project:
        raise ValueError(f"Project with key '{project_key}' not found.")

    is_creator = project["creator_id"] == member_id
    is_admin_member = role == "admin" and member_id in project["members"]

    if not (is_creator or is_admin_member):
        raise PermissionError("User not authorized to edit project settings.")

    project["settings"].update(updated_configuration)
    return f"Project '{project_key}' settings updated successfully."
