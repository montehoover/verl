# Global data structure for projects and their members
# This is a simplified example. In a real application, this might come from a database.
PROJECTS = {
    101: {"name": "Project Alpha", "members": [1, 2, 5]},
    102: {"name": "Project Beta", "members": [3, 4]},
    103: {"name": "Project Gamma", "members": [1, 4]},
    104: {"name": "Project Delta", "members": [1, 5, 2]},
}

def is_user_admin(user_id: int) -> int:
    """
    Counts how many active projects a user is a member of.

    Args:
        user_id: The ID of the user.

    Returns:
        The number of projects the user is a member of.
        Returns 0 if the user is not a member of any project.
    """
    project_count = 0
    for project_details in PROJECTS.values():
        if user_id in project_details.get("members", []):
            project_count += 1
    return project_count
