# Global data structure to track user roles
user_roles = {}

# Global data structure to track projects
PROJECTS = {}

def is_user_admin(user_id):
    """Count how many active projects a user is a member of.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        int: Number of projects the user is a member of
    """
    count = 0
    for project in PROJECTS.values():
        if user_id in project.get('members', []):
            count += 1
    return count

def add_user_to_project(project_id, user_id):
    """Add a user to a project.
    
    Args:
        project_id: The ID of the project
        user_id (int): The ID of the user to add
    """
    if project_id in PROJECTS:
        if 'members' not in PROJECTS[project_id]:
            PROJECTS[project_id]['members'] = []
        if user_id not in PROJECTS[project_id]['members']:
            PROJECTS[project_id]['members'].append(user_id)

def remove_user_from_project(project_id, user_id):
    """Remove a user from a project.
    
    Args:
        project_id: The ID of the project
        user_id (int): The ID of the user to remove
    """
    if project_id in PROJECTS and 'members' in PROJECTS[project_id]:
        if user_id in PROJECTS[project_id]['members']:
            PROJECTS[project_id]['members'].remove(user_id)
