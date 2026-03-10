# Placeholder for project data
PROJECTS_DB = {
    "proj1": {"name": "Project Alpha", "status": "active", "deadline": "2025-12-31"},
    "proj2": {"name": "Project Beta", "status": "pending", "deadline": "2026-06-30"},
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
    return PROJECTS_DB.get(proj_id)
