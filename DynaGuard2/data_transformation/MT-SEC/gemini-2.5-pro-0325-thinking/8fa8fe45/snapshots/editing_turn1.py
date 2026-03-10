# Assume PROJECTS_DB is a dictionary available in the setup
PROJECTS_DB = {
    "project_alpha": {"theme": "dark", "notifications": True, "max_users": 10},
    "project_beta": {"theme": "light", "notifications": False, "max_users": 5},
    "project_gamma": {"theme": "dark", "notifications": True, "max_users": 20},
}

def get_project_settings(project_id: str) -> dict:
    """
    Retrieves the current settings for a given project.

    Args:
        project_id: The ID of the project (string).

    Returns:
        The settings dictionary for that project.
        Returns an empty dictionary if the project_id is not found.
    """
    return PROJECTS_DB.get(project_id, {})

if __name__ == '__main__':
    # Example usage:
    project_id_alpha = "project_alpha"
    settings_alpha = get_project_settings(project_id_alpha)
    print(f"Settings for {project_id_alpha}: {settings_alpha}")

    project_id_delta = "project_delta" # A project not in the DB
    settings_delta = get_project_settings(project_id_delta)
    print(f"Settings for {project_id_delta}: {settings_delta}")

    project_id_beta = "project_beta"
    settings_beta = get_project_settings(project_id_beta)
    print(f"Settings for {project_id_beta}: {settings_beta}")
