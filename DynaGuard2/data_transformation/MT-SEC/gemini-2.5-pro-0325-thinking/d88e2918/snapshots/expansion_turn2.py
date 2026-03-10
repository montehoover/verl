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
