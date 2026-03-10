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
