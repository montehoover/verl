# Global data structure for projects
# Example: {project_id: {"name": "Project Name", "members": [user_id1, user_id2]}}
PROJECTS = {
    101: {"name": "Alpha Project", "members": [1, 2, 3]},
    102: {"name": "Beta Campaign", "members": [2, 4]},
    103: {"name": "Gamma Initiative", "members": [1, 4, 5]},
    104: {"name": "Delta Taskforce", "members": [3]},
}

def count_active_projects_for_user(user_id: int) -> int:
    """
    Counts how many active projects a user is a member of.

    Args:
        user_id: The ID of the user.

    Returns:
        The number of projects the user is a member of.
        Returns 0 if the user is not part of any project.
    """
    project_count = 0
    for project_details in PROJECTS.values():
        if user_id in project_details.get("members", []):
            project_count += 1
    return project_count
