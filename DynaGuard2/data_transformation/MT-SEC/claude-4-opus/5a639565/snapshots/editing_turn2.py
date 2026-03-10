# Global data structure to track projects
PROJECTS = {}

def is_user_admin(user_id):
    """Count how many active projects a user is a member of.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        int: The number of projects where the user appears in the 'members' list
    """
    count = 0
    for project in PROJECTS.values():
        if 'members' in project and user_id in project['members']:
            count += 1
    return count
