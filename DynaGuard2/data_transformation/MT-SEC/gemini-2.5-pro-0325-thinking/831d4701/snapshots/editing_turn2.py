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

# Sample PROJECT_MEMBERSHIP_DB dictionary
# Maps project_id to a list of user_ids who are members
PROJECT_MEMBERSHIP_DB = {
    "project_alpha": ["user_1", "user_2"],
    "project_beta": ["user_3"],
    "project_gamma": ["user_1", "user_4"],
}

def get_authorized_project_settings(user_id: str, project_id: str) -> dict | str:
    """
    Retrieves the settings for a project if the user is authorized.

    Args:
        user_id: The ID of the user requesting the settings.
        project_id: The ID of the project.

    Returns:
        A dictionary containing the project's settings if the user is a member,
        otherwise an error message string.
    """
    project_members = PROJECT_MEMBERSHIP_DB.get(project_id)
    if project_members and user_id in project_members:
        return get_project_settings(project_id)
    elif project_id not in PROJECTS_DB:
        return f"Error: Project '{project_id}' not found."
    else:
        return f"Error: User '{user_id}' is not authorized to access settings for project '{project_id}'."
