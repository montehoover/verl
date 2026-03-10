# Sample PROJECTS_DB dictionary
PROJECTS_DB = {
    "project_alpha": {
        "notifications": True,
        "theme": "dark",
        "auto_save": True,
    },
    "project_beta": {
        "notifications": False,
        "theme": "light",
        "auto_save": False,
    },
    "project_gamma": {
        "notifications": True,
        "theme": "dark",
        "auto_save": True,
        "experimental_features": ["feature_x", "feature_y"],
    },
}

def get_project_settings(project_id: str) -> dict:
    """
    Retrieves the settings dictionary for the specified project.

    Args:
        project_id: The ID of the project.

    Returns:
        A dictionary containing the project's settings, 
        or an empty dictionary if the project is not found.
    """
    return PROJECTS_DB.get(project_id, {})
