from typing import Union

PROJECTS_DB = {
    "project_alpha": {"notifications": True, "theme": "dark", "members": ["user1", "user2"]},
    "project_beta": {"notifications": False, "theme": "light", "members": ["user3"]},
    "project_gamma": {"notifications": True, "theme": "blue", "priority": "high", "members": ["user1", "user4"]},
}

def get_project_settings(project_id: str, user_id: str) -> Union[dict, str]:
    """
    Retrieves the current settings of a project if the user has access.

    Args:
        project_id: The ID of the project.
        user_id: The ID of the user requesting the settings.

    Returns:
        A dictionary containing the project settings if the user is a member,
        an error message string if the user lacks permissions,
        or an error message string if the project is not found.
    """
    project_data = PROJECTS_DB.get(project_id)
    if not project_data:
        return f"Error: Project '{project_id}' not found."

    if user_id not in project_data.get("members", []):
        return f"Error: User '{user_id}' does not have permission to access project '{project_id}' settings."

    # Return all data except 'members' list
    settings = {k: v for k, v in project_data.items() if k != "members"}
    return settings
