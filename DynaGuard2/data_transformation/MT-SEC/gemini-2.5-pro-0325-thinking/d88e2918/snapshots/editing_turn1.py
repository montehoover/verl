PROJECTS_DB = {
    "PROJ_A": {"theme": "dark", "notifications": True, "max_users": 10},
    "PROJ_B": {"theme": "light", "notifications": False, "max_users": 5},
    "PROJ_C": {"theme": "dark", "notifications": True, "max_users": 20, "experimental_features": False},
}

def get_project_settings(project_key: str) -> dict:
    """
    Retrieves the current settings of a project.

    Args:
        project_key: The key of the project (string).

    Returns:
        The settings dictionary for the specified project.
        Returns an empty dictionary if the project_key is not found.
    """
    return PROJECTS_DB.get(project_key, {})
