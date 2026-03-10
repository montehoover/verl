# Assume PROJECTS_DB is a dictionary available in the setup
PROJECTS_DB = {
    "project_alpha": {"theme": "dark", "notifications": True, "max_users": 10, "members": ["user1", "user2"]},
    "project_beta": {"theme": "light", "notifications": False, "max_users": 5, "members": ["user3"]},
    "project_gamma": {"theme": "dark", "notifications": True, "max_users": 20, "members": ["user1", "user4", "user5"]},
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

def is_member_of_project(user_id: str, project_id: str) -> bool:
    """
    Checks if a user is a member of a project.

    Args:
        user_id: The ID of the user (string).
        project_id: The ID of the project (string).

    Returns:
        True if the user is a member of the project, otherwise False.
    """
    project = PROJECTS_DB.get(project_id)
    if project and "members" in project:
        return user_id in project["members"]
    return False

if __name__ == '__main__':
    # Example usage for get_project_settings:
    project_id_alpha = "project_alpha"
    settings_alpha = get_project_settings(project_id_alpha)
    print(f"Settings for {project_id_alpha}: {settings_alpha}")

    project_id_delta = "project_delta" # A project not in the DB
    settings_delta = get_project_settings(project_id_delta)
    print(f"Settings for {project_id_delta}: {settings_delta}")

    project_id_beta = "project_beta"
    settings_beta = get_project_settings(project_id_beta)
    print(f"Settings for {project_id_beta}: {settings_beta}")

    # Example usage for is_member_of_project:
    user_id_1 = "user1"
    project_id_alpha_check = "project_alpha"
    is_member_alpha = is_member_of_project(user_id_1, project_id_alpha_check)
    print(f"Is {user_id_1} member of {project_id_alpha_check}? {is_member_alpha}")

    user_id_3 = "user3"
    project_id_beta_check = "project_beta"
    is_member_beta = is_member_of_project(user_id_3, project_id_beta_check)
    print(f"Is {user_id_3} member of {project_id_beta_check}? {is_member_beta}")

    user_id_non_member = "user_non_existent"
    is_member_gamma = is_member_of_project(user_id_non_member, "project_gamma")
    print(f"Is {user_id_non_member} member of project_gamma? {is_member_gamma}")

    is_member_non_project = is_member_of_project(user_id_1, "project_non_existent")
    print(f"Is {user_id_1} member of project_non_existent? {is_member_non_project}")
