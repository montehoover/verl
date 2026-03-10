# Placeholder for project data
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

def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on user_id and password.

    Args:
        user_id: The user's identifier.
        password: The user's password.

    Returns:
        True if the credentials are correct, False otherwise.
    """
    # This is a placeholder for actual authentication logic.
    # In a real application, you would check against a database
    # or an identity provider.
    # For demonstration, let's assume a hardcoded user.
    if user_id == "admin" and password == "password123":
        return True
    return False

def get_project_settings(proj_id: str) -> dict | None:
    """
    Retrieves the settings for a given project ID.

    Args:
        proj_id: The identifier of the project.

    Returns:
        A dictionary containing the project settings if found, otherwise None.
    """
    project = PROJECTS_DB.get(proj_id)
    if project:
        return project.get("settings")
    return None

def alter_project_preferences(member_id: str, member_role: str, proj_id: str, fresh_settings: dict) -> str:
    """
    Allows authorized users to modify project settings.

    Args:
        member_id: The ID of the user attempting the modification.
        member_role: The role of the user (e.g., 'admin', 'member').
        proj_id: The ID of the project whose settings are to be modified.
        fresh_settings: A dictionary containing the new settings to apply.

    Returns:
        A success message if the operation is successful.

    Raises:
        PermissionError: If the user is not authorized to modify the settings.
        ValueError: If the project ID is not found.
    """
    project = PROJECTS_DB.get(proj_id)

    if not project:
        raise ValueError(f"Project with ID '{proj_id}' not found.")

    is_creator = project.get("creator_id") == member_id
    is_admin = member_role == "admin"

    if not (is_creator or is_admin):
        raise PermissionError(f"User '{member_id}' is not authorized to modify settings for project '{proj_id}'.")

    # Update the settings
    project["settings"].update(fresh_settings)
    return f"Successfully updated settings for project '{proj_id}'."
