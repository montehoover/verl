# Global data structure to store project memberships
# Example: {project_id_1: {user_id_1, user_id_2}, project_id_2: {user_id_2, user_id_3}}
project_memberships = {}

def is_member_of_project(user_id: int, project_id: int) -> bool:
    """
    Checks if a specific user is a member of a given project.

    Args:
        user_id: The ID of the user.
        project_id: The ID of the project.

    Returns:
        True if the user is part of the project, otherwise False.
    """
    if project_id in project_memberships:
        return user_id in project_memberships[project_id]
    return False

def add_user_to_project(user_id: int, project_id: int) -> None:
    """
    Adds a user to a specific project.

    Args:
        user_id: The ID of the user.
        project_id: The ID of the project.
    """
    if project_id not in project_memberships:
        project_memberships[project_id] = set()
    project_memberships[project_id].add(user_id)

def remove_user_from_project(user_id: int, project_id: int) -> None:
    """
    Removes a user from a specific project.

    Args:
        user_id: The ID of the user.
        project_id: The ID of the project.
    """
    if project_id in project_memberships:
        project_memberships[project_id].discard(user_id)

def count_active_projects_for_user(user_id: int) -> int:
    """
    Counts how many active projects a user is a member of.

    Args:
        user_id: The ID of the user.

    Returns:
        The number of projects the user is a member of.
    """
    count = 0
    for project_id in project_memberships:
        if user_id in project_memberships[project_id]:
            count += 1
    return count
