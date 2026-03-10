# Global data structure for projects
# Example: {project_id: {"name": "Project Name", "members": [user_id1, user_id2]}}
PROJECTS = {
    101: {"name": "Alpha Project", "members": [1, 2, 5]},
    102: {"name": "Beta Campaign", "members": [2, 3]},
    103: {"name": "Gamma Initiative", "members": [1, 4]},
    104: {"name": "Delta Taskforce", "members": [1, 2, 3, 4]},
}

def count_active_user_projects(user_id: int) -> int:
    """
    Counts how many active projects a user is a member of.

    Args:
        user_id: The ID of the user.

    Returns:
        The number of projects the user is a member of.
        Returns 0 if the user isn't part of any project.
    """
    project_count = 0
    for project_details in PROJECTS.values():
        if user_id in project_details.get("members", []):
            project_count += 1
    return project_count
