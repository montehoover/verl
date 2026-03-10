PROJECTS_DB = {
    "PROJ_A": {"theme": "dark", "notifications": True, "max_users": 10},
    "PROJ_B": {"theme": "light", "notifications": False, "max_users": 5},
    "PROJ_C": {"theme": "dark", "notifications": True, "max_users": 20, "experimental_features": False},
}

PROJECT_MEMBERS_DB = {
    "PROJ_A": {"user1": "admin", "user2": "editor", "user3": "viewer"},
    "PROJ_B": {"user4": "admin", "user1": "viewer"},
    "PROJ_C": {"user5": "editor", "user2": "viewer", "user6": "admin"},
}

PRIVILEGED_ROLES = {"admin", "editor"}

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

def check_user_role(member_ID: str, project_key: str) -> bool:
    """
    Checks if a user has a role in a project that allows them to view or modify settings.

    Args:
        member_ID: The ID of the member (string).
        project_key: The key of the project (string).

    Returns:
        True if the user is a member of the project and has a privileged role,
        otherwise False.
    """
    project_members = PROJECT_MEMBERS_DB.get(project_key)
    if not project_members:
        return False
    
    member_role = project_members.get(member_ID)
    if not member_role:
        return False
        
    return member_role in PRIVILEGED_ROLES
