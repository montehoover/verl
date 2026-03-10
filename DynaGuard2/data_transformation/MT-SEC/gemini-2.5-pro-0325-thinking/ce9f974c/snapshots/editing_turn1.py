PROJECTS_DB = {
    "project_alpha": {"notifications": True, "theme": "dark"},
    "project_beta": {"notifications": False, "theme": "light"},
    "project_gamma": {"notifications": True, "theme": "blue", "priority": "high"},
}

def get_project_settings(project_id: str) -> dict:
    """
    Retrieves the current settings of a project.

    Args:
        project_id: The ID of the project.

    Returns:
        A dictionary containing the project settings,
        or an empty dictionary if the project is not found.
    """
    return PROJECTS_DB.get(project_id, {})
