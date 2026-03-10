# Global data structure for project memberships
project_memberships = {}

def is_member_of_project(user_id, project_id):
    """
    Check if a specific user is a member of a given project.
    
    Args:
        user_id (int): The ID of the user to check
        project_id (int): The ID of the project to check
        
    Returns:
        bool: True if the user is part of the project, otherwise False
    """
    if project_id in project_memberships:
        return user_id in project_memberships[project_id]
    return False
