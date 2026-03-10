# Global data structure for project memberships
PROJECTS = {}

def is_member_of_project(user_id, project_id):
    """
    Check if a specific user is a member of a given project.
    
    Args:
        user_id (int): The ID of the user to check
        project_id (int): The ID of the project to check
        
    Returns:
        bool: True if the user is part of the project, otherwise False
    """
    if project_id in PROJECTS:
        return user_id in PROJECTS[project_id].get('members', [])
    return False

def add_user_to_project(user_id, project_id):
    """
    Add a user to a project.
    
    Args:
        user_id (int): The ID of the user to add
        project_id (int): The ID of the project
    """
    if project_id not in PROJECTS:
        PROJECTS[project_id] = {'members': []}
    
    if user_id not in PROJECTS[project_id]['members']:
        PROJECTS[project_id]['members'].append(user_id)

def remove_user_from_project(user_id, project_id):
    """
    Remove a user from a project.
    
    Args:
        user_id (int): The ID of the user to remove
        project_id (int): The ID of the project
    """
    if project_id in PROJECTS and 'members' in PROJECTS[project_id]:
        if user_id in PROJECTS[project_id]['members']:
            PROJECTS[project_id]['members'].remove(user_id)

def is_user_admin(user_id):
    """
    Count how many active projects a user is a member of.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        int: The number of projects where this user appears in the 'members' list
    """
    count = 0
    for project_id, project_data in PROJECTS.items():
        if user_id in project_data.get('members', []):
            count += 1
    return count
